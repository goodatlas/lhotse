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

ZEROTH = ('recData01', 'recData02', 'recData03',
          'testData01', 'testData02')

# def download_librispeech(
#         target_dir: Pathlike = '.',
#         dataset_parts: Optional[Union[str, Sequence[str]]] = "mini_librispeech",
#         force_download: bool = False,
#         alignments: bool = False,
#         base_url: str = 'http://www.openslr.org/resources',
#         alignments_url: str = LIBRISPEECH_ALIGNMENTS_URL,
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

#     if dataset_parts == "librispeech":
#         dataset_parts = LIBRISPEECH
#     elif dataset_parts == "mini_librispeech":
#         dataset_parts = MINI_LIBRISPEECH
#     elif isinstance(dataset_parts, str):
#         dataset_parts = [dataset_parts]

#     for part in tqdm(dataset_parts, desc='Downloading LibriSpeech parts'):
#         logging.info(f'Processing split: {part}')
#         # Determine the valid URL for a given split.
#         if part in LIBRISPEECH:
#             url = f'{base_url}/12'
#         elif part in MINI_LIBRISPEECH:
#             url = f'{base_url}/31'
#         else:
#             logging.warning(f'Invalid dataset part name: {part}')
#             continue
#         # Split directory exists and seem valid? Skip this split.
#         part_dir = target_dir / f'LibriSpeech/{part}'
#         completed_detector = part_dir / '.completed'
#         if completed_detector.is_file():
#             logging.info(f'Skipping {part} because {completed_detector} exists.')
#             continue
#         # Maybe-download the archive.
#         tar_name = f'{part}.tar.gz'
#         tar_path = target_dir / tar_name
#         if force_download or not tar_path.is_file():
#             urlretrieve_progress(f'{url}/{tar_name}', filename=tar_path, desc=f'Downloading {tar_name}')
#         # Remove partial unpacked files, if any, and unpack everything.
#         shutil.rmtree(part_dir, ignore_errors=True)
#         with tarfile.open(tar_path) as tar:
#             tar.extractall(path=target_dir)
#         completed_detector.touch()

#     if alignments:
#         completed_detector = target_dir / '.ali_completed'
#         if completed_detector.is_file() and not force_download:
#             return
#         assert is_module_available('gdown'), 'To download LibriSpeech alignments, please install "pip install gdown"'
#         import gdown
#         ali_zip_path = str(target_dir / 'LibriSpeech-Alignments.zip')
#         gdown.download(alignments_url, output=ali_zip_path)
#         with zipfile.ZipFile(ali_zip_path) as f:
#             f.extractall(path=target_dir)
#             completed_detector.touch()


def prepare_zeroth(
    corpus_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = 'auto',
    output_dir: Optional[Pathlike] = None,
    # morpheme_analysis_model_path: Optional[Pathlike],
    # word_boundary_symbol: Optional[str],
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
                f"Could not find any of librispeech or mini_librispeech splits in: {corpus_dir}")
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

    speaker_info_path = corpus_dir / "SPEAKERS"
    if speaker_info_path.exists():
        speakers = {}
        print(f'found speaker_info file: {speaker_info_path}')
        with open(speaker_info_path) as f:
            for line in f:
                splits = line.strip().split('|')
                speaker_id = splits[0]
                speaker_name = splits[1]
                gender = splits[2]
                script_id = splits[3]
                speakers.setdefault(speaker_id, {})
                speakers[speaker_id]["name"] = speaker_name
                speakers[speaker_id]["gender"] = gender
                speakers[speaker_id]["script_id"] = script_id

    # if os.path.exists(morpheme_analysis_model_path):
    #    # load model
    #    print(
    #        f'Loading morpheme analysis model: {morpheme_analysis_model_path}')
    #    io = morfessor.MorfessorIO()
    #    model = io.read_binary_model_file(morpheme_analysis_model_path)

    #    print(f'word boundary symbol: {word_boundary_symbol}')

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
                            ex.submit(parse_utterance, part_path, line, speakers))

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
        # model: Optional[BaselineModel],
        # word_boundary_symbol: Optional[str]
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    recording_id, text = line.strip().split(maxsplit=1)
    # recording_id, text
    # 014_001_001 네 합정동이요 네 잠시만요 네 말씀해주세요 다진마늘 국산 하나 무 한포 세제 이 키로짜리 두 개 쇠고기 다시다
    # 014_001_002 저희는 상담사가요 열 명 정도 됩니다 사장님 배추 한 포 배추 아직까지 망으로 나와요 네 아직까지 망으로 나와요

    # if model is not None:
    #    smooth = 0
    #    maxlen = 30
    #    analyzed_text = ""
    #    for word in text.split():
    #        constructions, logp = model.viterbi_segment(
    #            word, smooth, maxlen)
    #        analyzed_text += word_boundary_symbol + \
    #            " ".join(constructions) + " "
    #    text = analyzed_text.strip()
    #    custom = "morpheme_updated"

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
        text=text.strip()
    )
    return recording, segment
