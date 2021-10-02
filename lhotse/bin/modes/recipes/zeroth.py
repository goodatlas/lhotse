import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.zeroth import prepare_zeroth
from lhotse.utils import Pathlike

__all__ = ['zeroth']


@prepare.command(context_settings=dict(show_default=True))
@click.argument('corpus_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('morpheme_analysis_model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_dir', type=click.Path())
@click.option('-j', '--num-jobs', type=int, default=1,
              help='How many threads to use (can give good speed-ups with slow disks).')
@click.option('-w', '--word-boundary-symbol', type=str, default="",
              help='word boundary symbol if it is defined in n-gram ARPA')
def zeroth(
        corpus_dir: Pathlike,
        output_dir: Pathlike,
        morpheme_analysis_model_path: Pathlike,
        word_boundary_symbol: str,
        num_jobs: int
):
    """Zeroth ASR data preparation."""
    prepare_zeroth(corpus_dir,
                   morpheme_analysis_model_path=morpheme_analysis_model_path,
                   word_boundary_symbol=word_boundary_symbol,
                   output_dir=output_dir, num_jobs=num_jobs)


# @download.command(context_settings=dict(show_default=True))
# @click.argument('target_dir', type=click.Path())
# @click.option('--full/--mini', default=True, help='Download Librispeech [default] or mini Librispeech.')
# Def librispeech(
#        target_dir: Pathlike,
#        full: bool
# ):
#    """(Mini) Librispeech download."""
#    download_librispeech(target_dir, dataset_parts='librispeech' if full else 'mini_librispeech')
