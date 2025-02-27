"""
This script creates the Room Impulse Response and Noises data directory.
This data includes all the room impulse responses (RIRs) and noises used in the 
paper "A Study on Data Augmentation of Reverberant Speech for Robust Speech Recognition" 
submitted to ICASSP 2017. It includes the real RIRs and isotropic noises from the RWCP 
sound scene database, the 2014 REVERB challenge database and the Aachen impulse response 
database (AIR); the simulated RIRs generated by ourselves and also the point-source 
noises that extracted from the MUSAN corpus.
The required dataset is freely available at http://www.openslr.org/17/

The dataset consists of 3 types of RIRs or noises:

1. pointsource_noises:

The point-source noises are sampled from the Freesound portion of the MUSAN corpus.
This portion of the corpus contains 843 noise recordings and each of them is manually 
classified as either a foreground or a background noise. The MUSAN corpus can be downloaded by
wget http://www.openslr.org/resources/17/musan.tar.gz


2. real_rirs_isotropic_noises:

The set of real RIRs is composed of three databases: the RWCP sound scene database, 
the 2014 REVERB challenge database and the Aachen impulse response database (AIR).
Overall there are 325 real RIRs. The isotropic noises available in the real RIR databases are
used along with the associated RIRs. Here are the links to download the individual databases:

  - AIR `wget http://www.openslr.org/resources/20/air_database_release_1_4.zip`
  - RWCP `wget http://www.openslr.org/resources/13/RWCP.tar.gz`
  - 2014 REVERB challenge,
    `wget http://reverb2014.dereverberation.com/tools/reverb_tools_for_Generate_mcTrainData.tgz`
    `wget http://reverb2014.dereverberation.com/tools/reverb_tools_for_Generate_SimData.tgz`


3. simulated_rirs

This folder contains the simulated RIRs. Please go to simulated_rirs/README for the details of the data.
This simulated RIR data set can also be downloaded from wget http://www.openslr.org/resources/26/sim_rir.zip

The corpus can be cited as follows:
@article{Ko2017ASO,
  title={A study on data augmentation of reverberant speech for robust speech recognition},
  author={Tom Ko and Vijayaditya Peddinti and Daniel Povey and Michael L. Seltzer and Sanjeev Khudanpur},
  journal={2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2017},
  pages={5220-5224}
}
"""
from collections import defaultdict
import logging
import zipfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from lhotse import Recording, RecordingSet, CutSet
from lhotse.audio import AudioSource
from lhotse.utils import Pathlike, urlretrieve_progress

RIR_NOISE_ZIP_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"

PARTS = {
    "point_noise": "pointsource_noises",
    "iso_noise": "real_rirs_isotropic_noises",
    "real_rir": "real_rirs_isotropic_noises",
    "sim_rir": "simulated_rirs",
}


def download_rir_noise(
    target_dir: Pathlike = ".",
    url: Optional[str] = RIR_NOISE_ZIP_URL,
    force_download: Optional[bool] = False,
) -> None:
    """
    Download and untar the RIR Noise corpus.

    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param url: str, the url that downloads file called "rirs_noises.zip".
    :param force_download: bool, if True, download the archive even if it already exists.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "rirs_noises.zip"
    zip_path = target_dir / zip_name
    if zip_path.exists() and not force_download:
        logging.info(f"Skipping {zip_name} because file exists.")
    else:
        urlretrieve_progress(url, zip_path, desc=f"Downloading {zip_name}")
        logging.info(f"Downloaded {zip_name}.")
    zip_dir = target_dir / "rirs_noises"
    if not zip_dir.exists():
        logging.info(f"Unzipping {zip_name}.")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)


def prepare_rir_noise(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    parts: Sequence[str] = ("point_noise", "iso_noise", "real_rir", "sim_rir"),
) -> Dict[str, Dict[str, Union[RecordingSet, CutSet]]]:
    """
    Prepare the RIR Noise corpus.

    :param corpus_dir: Pathlike, the path of the dir to store the dataset.
    :param output_dir: Pathlike, the path of the dir to write the manifests.
    :param parts: Sequence[str], the parts of the dataset to prepare.

    The corpus contains 4 things: point-source noises (point_noise), isotropic noises (iso_noise),
    real RIRs (real_rir), and simulated RIRs (sim_rir). We will prepare these parts
    in the corresponding dict keys.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if not parts:
        raise ValueError("No parts specified for manifest preparation.")
    if isinstance(parts, str):
        parts = [parts]

    manifests = defaultdict(dict)
    for part in parts:
        logging.info(f"Preparing {part}...")
        audio_dir = corpus_dir / PARTS[part]
        assert audio_dir.is_dir(), f"No such directory: {audio_dir}"
        if part == "sim_rir":
            # The "small", "medium", and "large" rooms have the same file names, so
            # we have to handle them separately to avoid duplicating manifests.
            recordings = []
            for room_type in ("small", "medium", "large"):
                room_dir = audio_dir / f"{room_type}room"
                recordings += [
                    Recording.from_file(file, recording_id=f"{room_type}-{file.stem}")
                    for file in room_dir.rglob("*.wav")
                ]
            manifests[part]["recordings"] = RecordingSet.from_recordings(recordings)
        elif part == "point_noise":
            manifests[part]["recordings"] = RecordingSet.from_recordings(
                Recording.from_file(file) for file in audio_dir.rglob("*.wav")
            )
        elif part == "iso_noise":
            manifests[part]["recordings"] = RecordingSet.from_recordings(
                Recording.from_file(file)
                for file in audio_dir.rglob("*.wav")
                if "noise" in file.stem
            )
        elif part == "real_rir":
            manifests[part]["recordings"] = RecordingSet.from_recordings(
                Recording.from_file(file)
                for file in audio_dir.rglob("*.wav")
                if "rir" in file.stem
            )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for part in manifests:
            for key, manifest in manifests[part].items():
                manifest.to_file(output_dir / f"{key}_{part}.json")

    return manifests
