import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.ksponspeech import prepare_ksponspeech
from lhotse.utils import Pathlike

__all__ = ['ksponspeech']


@prepare.command(context_settings=dict(show_default=True))
@click.argument('corpus_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
@click.option('-m', '--morpheme-analysis-model_path', type=click.Path(), default="")
@click.option('-j', '--num-jobs', type=int, default=1,
              help='How many threads to use (can give good speed-ups with slow disks).')
@click.option('-w', '--word-boundary-symbol', type=str, default="",
              help='word boundary symbol if it is defined in n-gram ARPA')
def ksponspeech(
        corpus_dir: Pathlike,
        output_dir: Pathlike,
        morpheme_analysis_model_path: Pathlike,
        word_boundary_symbol: str,
        num_jobs: int
):
    """KsponSpeech AIHub data preparation."""
    prepare_ksponspeech(corpus_dir,
                        morpheme_analysis_model_path=morpheme_analysis_model_path,
                        word_boundary_symbol=word_boundary_symbol,
                        output_dir=output_dir, num_jobs=num_jobs)


# @download.command(context_settings=dict(show_default=True))
# @click.argument('target_dir', type=click.Path())
# def zeroth(
#     target_dir: Pathlike,
# ):
#     """Zeroth data download"""
#     download_zerothspeech(target_dir)
