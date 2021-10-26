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

# based on openspeech/datasets/ksponspeech/preprocess/preprocess.py
# https://github.com/openspeech-team/openspeech/blob/main/openspeech/datasets/ksponspeech/preprocess/preprocess.py


def bracket_filter(sentence, mode='phonetic'):
    new_sentence = str()

    if mode == 'phonetic':
        flag = False

        for ch in sentence:
            if ch == '(' and flag is False:
                flag = True
                continue
            if ch == '(' and flag is True:
                flag = False
                continue
            if ch != ')' and flag is False:
                new_sentence += ch

    elif mode == 'spelling':
        flag = True

        for ch in sentence:
            if ch == '(':
                continue
            if ch == ')':
                if flag is True:
                    flag = False
                    continue
                else:
                    flag = True
                    continue
            if ch != ')' and flag is True:
                new_sentence += ch

    else:
        raise ValueError("Unsupported mode : {0}".format(mode))

    return new_sentence


def special_filter(sentence, mode='phonetic', replace=None):
    SENTENCE_MARK = ['?', '!', '.']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['/', '+', '*', '-', '@', '$',
              '^', '&', '[', ']', '=', ':', ';', ',']

    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
                continue

        if ch == '#':
            new_sentence += '샾'

        elif ch == '%':
            if mode == 'phonetic':
                new_sentence += replace
            elif mode == 'spelling':
                new_sentence += '%'

        elif ch not in EXCEPT:
            new_sentence += ch

    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence


def sentence_filter(raw_sentence, mode, replace=None):
    return special_filter(bracket_filter(raw_sentence, mode), mode, replace)


PERCENT_FILES = {
    '087797': '퍼센트',
    '215401': '퍼센트',
    '284574': '퍼센트',
    '397184': '퍼센트',
    '501006': '프로',
    '502173': '프로',
    '542363': '프로',
    '581483': '퍼센트'
}


def read_preprocess_text_file(file_path, mode):
    # with open(file_path, 'r', encoding='cp949') as f:
    with open(file_path, 'r') as f:
        raw_sentence = f.read()
        file_name = os.path.basename(file_path)
        if file_name[12:18] in PERCENT_FILES.keys():
            replace = PERCENT_FILES[file_name[12:18]]
        else:
            replace = None
        return sentence_filter(raw_sentence, mode=mode, replace=replace)


KSPONSPEECH = ('KsponSpeech_01', 'KsponSpeech_02', 'KsponSpeech_03',
               'KsponSpeech_04', 'KsponSpeech_05')


# class ProgressPercentage(object):
#     ''' Progress Class
#     Class for calculating and displaying download progress
#     '''
#     # https://gist.github.com/egeulgen/538aadc90275d79d514a5bacc4d5694e

#     def __init__(self, client, bucket, filename):
#         ''' Initialize
#         initialize with: file name, file size and lock.
#         Set seen_so_far to 0. Set progress bar length
#         '''
#         self._filename = filename
#         self._size = client.head_object(Bucket=bucket, Key=filename)[
#             'ContentLength']
#         self._seen_so_far = 0
#         self._lock = threading.Lock()
#         self.prog_bar_len = 80

#     def __call__(self, bytes_amount):
#         ''' Call
#         When called, increments seen_so_far by bytes_amount,
#         calculates percentage of seen_so_far/total file size
#         and prints progress bar.
#         '''
#         # To simplify we'll assume this is hooked up to a single filename.
#         with self._lock:
#             self._seen_so_far += bytes_amount
#             ratio = round((float(self._seen_so_far) /
#                           float(self._size)) * (self.prog_bar_len - 6), 1)
#             current_length = int(round(ratio))

#             percentage = round(100 * ratio / (self.prog_bar_len - 6), 1)

#             bars = '+' * current_length
#             output = bars + ' ' * (self.prog_bar_len - current_length -
#                                    len(str(percentage)) - 1) + str(percentage) + '%'

#             if self._seen_so_far != self._size:
#                 sys.stdout.write(output + '\r')
#             else:
#                 sys.stdout.write(output + '\n')
#             sys.stdout.flush()


# class ZerothSpeechDownloader():
#     """ZerothSpeechDownloader
#     Class for downloading files from AWS
#     """

#     def __init__(self,
#                  aws_access_key_id=None,
#                  aws_secret_access_key=None,
#                  region_name=None):
#         """
#         normally aws credential should be set through `aws configure` by aws cli
#         """

#         self.client = boto3.client('s3')
#         if aws_access_key_id is not None:
#             self.client = boto3.client('s3',
#                                        aws_access_key_id=aws_access_key_id,
#                                        aws_secret_access_key=aws_secret_access_key,
#                                        region_name=region_name)
#         self.bucket_name = 'zeroth-opensource'

#     def download(self, filename, dest_name):
#         s3 = boto3.resource('s3')
#         progress = ProgressPercentage(self.client, self.bucket_name, filename)
#         s3.Bucket(self.bucket_name).download_file(
#             filename, dest_name, Callback=progress)


# def download_zerothspeech(
#         target_dir: Pathlike = '.',
#         # dataset_parts: Optional[Union[str, Sequence[str]]] = "mini_librispeech",
#         force_download: bool = False,
# ) -> None:
#     """
#     Download and untar the dataset, supporting both LibriSpeech and MiniLibrispeech

#     :param target_dir: Pathlike, the path of the dir to storage the dataset.
#     :param dataset_parts: "librispeech", "mini_librispeech",
#         or a list of splits (e.g. "dev-clean") to download.
#     :param force_download: Bool, if True, download the tars no matter if the tars exist.
#     :param alignments: should we download the alignments. The original source is:
#         https://github.com/CorentinJ/librispeech-alignments
#     :param base_url: str, the url of the OpenSLR resources.
#     :param alignments_url: str, the url of LibriSpeech word alignments
#     """
#     target_dir = Path(target_dir)
#     target_dir.mkdir(parents=True, exist_ok=True)

#     dataset_parts = ZEROTH

#     zeroth_dl = ZerothSpeechDownloader()

#     for part in tqdm(dataset_parts, desc='Downloading ZerothSpeech parts'):
#         logging.info(f'Processing split: {part}')

#         # Split directory exists and seem valid? Skip this split.
#         part_dir = target_dir / f'zeroth/{part}'
#         part_dir.mkdir(parents=True, exist_ok=True)
#         completed_detector = part_dir / '.completed'
#         if completed_detector.is_file():
#             logging.info(
#                 f'Skipping {part} because {completed_detector} exists.')
#             continue
#         # Maybe-download the archive.
#         tar_name = f'{part}.tar.gz'
#         tar_path = target_dir / tar_name
#         if force_download or not tar_path.is_file():
#             zeroth_dl.download('v2/' + str(tar_name), str(tar_path))
#         # Remove partial unpacked files, if any, and unpack everything.
#         shutil.rmtree(part_dir, ignore_errors=True)
#         with tarfile.open(tar_path) as tar:
#             tar.extractall(path=part_dir)
#         completed_detector.touch()

#     for part in tqdm(('AUDIO_INFO', 'README.md'), desc='Downloading ZerothSpeech info'):
#         logging.info(f'Processing info: {part}')

#         # file exists and seem valid? Skip this file.
#         target_file = target_dir / f'zeroth/{part}'
#         if target_file.is_file():
#             logging.info(
#                 f'Skipping {part} because {target_file} exists.')
#             continue
#         # Maybe-download the archive.
#         tar_name = f'{part}'
#         tar_path = target_file.parent / tar_name
#         if force_download or not tar_path.is_file():
#             zeroth_dl.download('v2/' + str(tar_name), str(tar_path))

def prepare_ksponspeech(
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
            set(KSPONSPEECH)
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

    if os.path.exists(morpheme_analysis_model_path):
        # load model
        print(
            f'Loading morpheme analysis model: {morpheme_analysis_model_path}')
        io = morfessor.MorfessorIO()
        model = io.read_binary_model_file(morpheme_analysis_model_path)

        print(f'word boundary symbol: {word_boundary_symbol}')

    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc='Dataset parts'):
            logging.info(f'Processing KsponSpeech subset: {part}')
            if manifests_exist(part=part, output_dir=output_dir):
                logging.info(
                    f'KsponSpeech subset: {part} already prepared - skipping.')
                continue
            recordings = []
            supervisions = []
            part_path = corpus_dir / part
            futures = []
            for trans_path in tqdm(part_path.rglob('*.utf8'), desc='Distributing tasks', leave=False):
                # corpus/KsponSpeech_01/KsponSpeech_0001
                #   KsponSpeech_000198.wav
                #   KsponSpeech_000198.utf8
                # with open(trans_path) as f:
                #    raw_sentence = f.read()
                futures.append(ex.submit(parse_utterance,
                               trans_path, model, word_boundary_symbol))

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
        trans_path: Path,
        model: Optional[BaselineModel],
        word_boundary_symbol: Optional[str]
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    # trans_path: AIhub/KsponSpeech_01/KsponSpeech_0016/KsponSpeech_015435.utf8
    # line: o/ 짐 적금 깨가지고 돈 다 쓰고 b/ (50만 원)/(오십만 원) 남아서 b/ (50만 원)/(오십만 원) 다시 적금했거든. b/ 아/ 이것도 깨고+ 깨게 생겼어 지금. b/
    trans_path = str(trans_path)

    # normalization from openspeech
    mode = 'phonetic'
    text = read_preprocess_text_file(trans_path, mode)

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
    file_path, ext = trans_path.strip().split('.', maxsplit=1)
    file_id = os.path.basename((file_path))
    if ext != "utf8":
        logging.warning(f'extension extraction failed on {trans_path}')
        return None

    audio_path = Path(file_path + '.wav')
    if not audio_path.is_file():
        logging.warning(f'No such file: {audio_path}')
        return None
    recording = Recording.from_file(audio_path, recording_id=file_id)

    # Then, create the corresponding supervisions
    segment = SupervisionSegment(
        id=file_id,
        recording_id=file_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language='Korean',
        speaker=None,
        gender=None,
        custom=None if custom is None else custom,
        text=text.strip()
    )
    return recording, segment
