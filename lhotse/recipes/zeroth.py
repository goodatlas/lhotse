import threading
import sys
import boto3
import logging
import re
import shutil
import tarfile
import zipfile
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

import morfessor
from morfessor import BaselineModel
import os

ZEROTH = ('train', 'dev', 'test')


class ProgressPercentage(object):
    ''' Progress Class
    Class for calculating and displaying download progress
    '''
    # https://gist.github.com/egeulgen/538aadc90275d79d514a5bacc4d5694e

    def __init__(self, client, bucket, filename):
        ''' Initialize
        initialize with: file name, file size and lock.
        Set seen_so_far to 0. Set progress bar length
        '''
        self._filename = filename
        self._size = client.head_object(Bucket=bucket, Key=filename)[
            'ContentLength']
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self.prog_bar_len = 80

    def __call__(self, bytes_amount):
        ''' Call
        When called, increments seen_so_far by bytes_amount,
        calculates percentage of seen_so_far/total file size
        and prints progress bar.
        '''
        # To simplify we'll assume this is hooked up to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            ratio = round((float(self._seen_so_far) /
                          float(self._size)) * (self.prog_bar_len - 6), 1)
            current_length = int(round(ratio))

            percentage = round(100 * ratio / (self.prog_bar_len - 6), 1)

            bars = '+' * current_length
            output = bars + ' ' * (self.prog_bar_len - current_length -
                                   len(str(percentage)) - 1) + str(percentage) + '%'

            if self._seen_so_far != self._size:
                sys.stdout.write(output + '\r')
            else:
                sys.stdout.write(output + '\n')
            sys.stdout.flush()


class ZerothSpeechDownloader():
    """ZerothSpeechDownloader
    Class for downloading files from AWS
    """

    def __init__(self,
                 aws_access_key_id="AKIASLNRD65N3ETEFPJW",
                 aws_secret_access_key="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",
                 region_name="ap-northeast-2"):

        self.client = boto3.client('s3',
                                   aws_access_key_id=aws_access_key_id,
                                   aws_secret_access_key=aws_secret_access_key,
                                   region_name=region_name)
        self.bucket_name = 'zeroth-opensource'

    def download(self, filename, dest_name):
        s3 = boto3.resource('s3')
        progress = ProgressPercentage(self.client, self.bucket_name, filename)
        s3.Bucket(self.bucket_name).download_file(
            filename, dest_name, Callback=progress)


def download_zerothspeech(
        target_dir: Pathlike = '.',
        # dataset_parts: Optional[Union[str, Sequence[str]]] = "mini_librispeech",
        force_download: bool = False,
) -> None:
    """
    Download and untar the dataset, supporting both LibriSpeech and MiniLibrispeech

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param dataset_parts: "librispeech", "mini_librispeech",
        or a list of splits (e.g. "dev-clean") to download.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param alignments: should we download the alignments. The original source is:
        https://github.com/CorentinJ/librispeech-alignments
    :param base_url: str, the url of the OpenSLR resources.
    :param alignments_url: str, the url of LibriSpeech word alignments
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset_parts = ZEROTH

    zeroth_dl = ZerothSpeechDownloader()

    for part in tqdm(dataset_parts, desc='Downloading ZerothSpeech parts'):
        logging.info(f'Processing split: {part}')

        # Split directory exists and seem valid? Skip this split.
        part_dir = target_dir / f'zeroth/{part}'
        part_dir.mkdir(parents=True, exist_ok=True)
        completed_detector = part_dir / '.completed'
        if completed_detector.is_file():
            logging.info(
                f'Skipping {part} because {completed_detector} exists.')
            continue
        # Maybe-download the archive.
        tar_name = f'{part}.tar.gz'
        tar_path = target_dir / tar_name
        if force_download or not tar_path.is_file():
            zeroth_dl.download('v2/' + str(tar_name), str(tar_path))
        # Remove partial unpacked files, if any, and unpack everything.
        shutil.rmtree(part_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=part_dir)
        completed_detector.touch()

    for part in tqdm(('AUDIO_INFO', 'README.md'), desc='Downloading ZerothSpeech info'):
        logging.info(f'Processing info: {part}')

        # file exists and seem valid? Skip this file.
        target_file = target_dir / f'zeroth/{part}'
        if target_file.is_file():
            logging.info(
                f'Skipping {part} because {target_file} exists.')
            continue
        # Maybe-download the archive.
        tar_name = f'{part}'
        tar_path = target_file.parent / tar_name
        if force_download or not tar_path.is_file():
            zeroth_dl.download('v2/' + str(tar_name), str(tar_path))


def prepare_zeroth(
    corpus_dir: Pathlike,
    morpheme_analysis_model_path: Optional[Pathlike],
    word_boundary_symbol: Optional[str],
    dataset_parts: Union[str, Sequence[str]] = 'auto',
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f'No such directory: {corpus_dir}'

    if dataset_parts == 'auto':
        dataset_parts = (
            set(ZEROTH)
            .intersection(path.name for path in corpus_dir.glob('*'))
        )
        if not dataset_parts:
            raise ValueError(
                f"Could not find any of zeroth splits in: {corpus_dir}")
    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

    print(f"dataset_parts:  {dataset_parts}")

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(dataset_parts=dataset_parts,
                                             output_dir=output_dir)

    speaker_info_path = corpus_dir / "AUDIO_INFO"
    if speaker_info_path.exists():
        speakers = {}
        print(f'found speaker_info file: {speaker_info_path}')
        with open(speaker_info_path) as f:
            for line in f:
                splits = line.strip().split('|')
                if len(splits) <= 1:
                    logging.debug(f"unexpected line: {line}")
                    continue
                speaker_id = splits[0]
                speaker_name = splits[1]
                gender = splits[2]
                script_id = splits[3]
                speakers.setdefault(speaker_id, {})
                speakers[speaker_id]["name"] = speaker_name
                speakers[speaker_id]["gender"] = gender
                speakers[speaker_id]["script_id"] = script_id

    if os.path.exists(morpheme_analysis_model_path):
        # load model
        print(
            f'Loading morpheme analysis model: {morpheme_analysis_model_path}')
        io = morfessor.MorfessorIO()
        model = io.read_binary_model_file(morpheme_analysis_model_path)

        print(f'word boundary symbol: {word_boundary_symbol}')

    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc='Dataset parts'):
            logging.info(f'Processing zeroth subset: {part}')
            if manifests_exist(part=part, output_dir=output_dir):
                logging.info(
                    f'zeroth subset: {part} already prepared - skipping.')
                continue
            recordings = []
            supervisions = []
            part_path = corpus_dir / part  # corpus/recData01
            futures = []
            for trans_path in tqdm(part_path.rglob('*.txt'), desc='Distributing tasks', leave=False):
                # corpus/recData01/001/014
                #   014_001.trans.txt
                #   014_001_001.flac
                with open(trans_path) as f:
                    for line in f:
                        futures.append(
                            ex.submit(parse_utterance, part_path, line, speakers, model, word_boundary_symbol))

            for future in tqdm(futures, desc='Processing', leave=False):
                result = future.result()
                if result is None:
                    continue
                recording, segment = result
                recordings.append(recording)
                supervisions.append(segment)

            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)

            validate_recordings_and_supervisions(
                recording_set, supervision_set)

            if output_dir is not None:
                supervision_set.to_json(
                    output_dir / f'supervisions_{part}.json')
                recording_set.to_json(output_dir / f'recordings_{part}.json')

            manifests[part] = {
                'recordings': recording_set,
                'supervisions': supervision_set
            }

    return dict(manifests)  # Convert to normal dict

    # return import_kaldi_data_dir(data_dir, dataset_parts, output_dir)


def parse_utterance(
        dataset_split_path: Path,
        line: str,
        speakers: Optional[Dict],
        model: Optional[BaselineModel],
        word_boundary_symbol: Optional[str]
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    recording_id, text = line.strip().split(maxsplit=1)
    # recording_id, text
    # 014_001_001 네 합정동이요 네 잠시만요 네 말씀해주세요 다진마늘 국산 하나 무 한포 세제 이 키로짜리 두 개 쇠고기 다시다
    # 014_001_002 저희는 상담사가요 열 명 정도 됩니다 사장님 배추 한 포 배추 아직까지 망으로 나와요 네 아직까지 망으로 나와요

    # exclude setencese with non-Korean characters
    match = re.search(r"[^가-힣\s]+", text)
    if match:
        return None

    # apply morpheme analysis on word-based text
    if model is not None:
        smooth = 0
        maxlen = 30
        analyzed_text = ""
        for word in text.split():
            constructions, logp = model.viterbi_segment(
                word, smooth, maxlen)
            analyzed_text += word_boundary_symbol + \
                " ".join(constructions) + " "
        text = analyzed_text.strip()
        custom = "morpheme_updated"

    # Create the Recording first
    speaker_id = recording_id.split('_')[0]
    script_id = recording_id.split('_')[1]
    utt_id = recording_id.split('_')[2]

    audio_path = dataset_split_path / \
        Path(script_id + "/" + speaker_id + "/" +
             utt_id).parent / f'{recording_id}.flac'
    if not audio_path.is_file():
        logging.warning(f'No such file: {audio_path}')
        return None
    recording = Recording.from_file(audio_path, recording_id=recording_id)

    # Then, create the corresponding supervisions
    if speakers is not None:
        speaker = speakers[speaker_id]["name"]
        gender = speakers[speaker_id]["gender"]

    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language='Korean',
        speaker=re.sub(
            r'-.*', r'', recording.id) if speaker is None else speaker,
        gender=None if gender is None else gender,
        custom=None if custom is None else custom,
        text=text.strip()
    )
    return recording, segment
