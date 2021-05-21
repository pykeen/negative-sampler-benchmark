"""Main script."""

import getpass
import logging
import math
import pathlib
from typing import Iterable, Optional, cast

import click
import pandas as pd
import seaborn as sns
import torch
from docdata import get_docdata
from matplotlib import pyplot as plt
from more_click import force_option, verbose_option
from torch.utils.benchmark import Timer
from tqdm import tqdm

import pykeen.version
from pykeen.datasets import datasets as datasets_dict, get_dataset
from pykeen.datasets.base import Dataset
from pykeen.sampling import negative_sampler_resolver
from pykeen.sampling.filtering import PythonSetFilterer, filterer_resolver

logger = logging.getLogger(__name__)

HERE = pathlib.Path(__file__).resolve().parent
measurement_root = HERE.joinpath("data")
plot_root = HERE.joinpath("img")

#: Datasets to benchmark. Only pick pre-stratified ones
_datasets = [
    'kinships',
    'nations',
    'umls',
    'countries',
    'codexsmall',
    'codexmedium',
    'codexlarge',
    'fb15k',
    'fb15k237',
    'wn18',
    'wn18rr',
    'yago310',
    'dbpedia50',
]
# Order by increasing number of triples
_datasets = sorted(_datasets, key=lambda s: get_docdata(datasets_dict[s])['statistics']['triples'])

USER = getpass.getuser()

dataset_option = click.option("-d", "--dataset", type=click.Choice(datasets_dict, case_sensitive=False))
directory_option = click.option("-o", "--directory", type=pathlib.Path, default=measurement_root)
plot_directory_option = click.option("--plot-directory", type=pathlib.Path, default=plot_root)
no_hashed_option = click.option("--no-hashed", is_flag=True)

sns.set_style('whitegrid')


@click.group()
def benchmark():
    """Benchmark negative sampling."""


@benchmark.command()
@dataset_option
@click.option("-b", "--num-random-batches", type=int, default=20)
@directory_option
@plot_directory_option
@force_option
@no_hashed_option
@verbose_option
def time(
    dataset: str,
    num_random_batches: int,
    directory: pathlib.Path,
    plot_directory: pathlib.Path,
    force: bool,
    no_hashed: bool,
) -> None:
    """Benchmark sampling time."""
    key = 'times'
    directory = _prep_dir(directory, key=key, hashed=not no_hashed)
    plot_directory = _prep_dir(plot_directory, hashed=not no_hashed)

    dfs = []
    for dataset_instance in _iterate_datasets(dataset):
        dfs.append(_time_helper(
            dataset_instance,
            directory=directory,
            num_random_batches=num_random_batches,
            force=force,
        ))
    _plot_times(pd.concat(dfs), key=key, directory=plot_directory)


def _time_helper(
    dataset: Dataset,
    *,
    directory: pathlib.Path,
    num_random_batches: int,
    force: bool = False,
) -> pd.DataFrame:
    output_path = directory.joinpath(dataset.get_normalized_name()).with_suffix('.tsv')
    if output_path.exists() and not force:
        tqdm.write(f'Using pre-calculated time results for {dataset.get_normalized_name()}')
        return pd.read_csv(output_path, sep='\t')

    batch_sizes = [2 ** i for i in range(1, min(16, int(math.log2(dataset.training.num_triples))))]
    logger.info(f"Evaluating batch sizes {batch_sizes} for dataset {dataset.get_normalized_name()}")

    data = []
    for negative_sampler in negative_sampler_resolver.lookup_dict.keys():
        sampler = negative_sampler_resolver.make(query=negative_sampler, triples_factory=dataset.training)
        progress = tqdm(
            (
                (b, i)
                for b in batch_sizes
                for i in range(num_random_batches)
            ),
            unit_scale=True,
            desc=f"Time {negative_sampler}",
            total=len(batch_sizes) * num_random_batches,
        )

        for batch_size, batch_id in progress:
            progress.set_postfix(batch_size=batch_size, dataset=dataset.get_normalized_name())
            positive_batch_idx = torch.randperm(dataset.training.num_triples)[:batch_size]
            positive_batch = dataset.training.mapped_triples[positive_batch_idx]
            timer = Timer(
                stmt="sampler.corrupt_batch(positive_batch=positive_batch)",
                globals=dict(
                    sampler=sampler,
                    positive_batch=positive_batch,
                ),
            )
            measurement = timer.blocked_autorange()
            data.extend(
                (
                    USER,
                    pykeen.version.get_git_hash(),
                    dataset.get_normalized_name(),
                    negative_sampler,
                    batch_size,
                    batch_id,
                    t,
                )
                for t in measurement.raw_times
            )
    df = pd.DataFrame(data=data, columns=["user", "hash", "dataset", "sampler", "batch_size", "batch_id", "time"])
    df.to_csv(output_path, sep="\t", index=False)
    return df


def _plot_times(df: pd.DataFrame, *, key: str, directory: pathlib.Path = plot_root):
    g = sns.relplot(
        data=df,
        x="batch_size",
        y="time",
        hue="sampler",
        kind="line",
        col="dataset",
        height=3.5,
        col_wrap=4 if df.dataset.nunique() > 4 else None,
        # ci=100,
        # estimator=numpy.median,
    ).set(
        xscale="log",
        xlabel="batch size",
        ylabel="time (s)/batch",
    )
    g.tight_layout()
    g.fig.suptitle(_prep_title('Time Results'), fontsize=22, y=0.98)
    make_space_above(g.axes, topmargin=1.0)

    directory.mkdir(exist_ok=True, parents=True)
    figure_path_stem = directory.joinpath(key)
    plt.savefig(figure_path_stem.with_suffix('.svg'))
    plt.savefig(figure_path_stem.with_suffix('.pdf'))
    plt.savefig(figure_path_stem.with_suffix('.png'), dpi=300)


@benchmark.command()
@dataset_option
@click.option("-b", "--batch_size", type=int, default=32)
@click.option("-n", "--num-samples", type=int, default=4096)  # TODO: Determine this based on dataset?
@directory_option
@plot_directory_option
@verbose_option
@no_hashed_option
@force_option
def fnr(
    dataset: str,
    batch_size: int,
    num_samples: int,
    directory: pathlib.Path,
    plot_directory: pathlib.Path,
    force: bool,
    no_hashed: bool,
):
    """Estimate false negative rate."""
    key = 'fnr'
    directory = _prep_dir(directory, key=key, hashed=not no_hashed)
    plot_directory = _prep_dir(plot_directory, hashed=not no_hashed)

    dfs = []
    for dataset_instance in _iterate_datasets(dataset):
        dfs.append(_fnr_helper(
            dataset_instance,
            directory=directory,
            num_samples=num_samples,
            batch_size=batch_size,
            force=force,
        ))
    _plot_fnr(pd.concat(dfs), directory=plot_directory, key=key)


def _fnr_helper(
    dataset: Dataset,
    *,
    directory: pathlib.Path,
    num_samples: int,
    batch_size: int,
    force: bool = False,
) -> pd.DataFrame:
    output_path = directory.joinpath(dataset.get_normalized_name()).with_suffix('.tsv')
    if output_path.exists() and not force:
        click.secho(f'Using pre-calculated fnr results for {dataset.get_normalized_name()}')
        return pd.read_csv(output_path, sep='\t')

    # create index structure for existence check
    filterer = PythonSetFilterer(
        mapped_triples=cast(torch.LongTensor, torch.cat(
            [
                dataset.training.mapped_triples,
                dataset.validation.mapped_triples,
                # dataset_instance.testing, # TODO: should this be used?
            ],
            dim=0,
        )),
    )
    filterer_key = filterer_resolver.normalize_inst(filterer)
    data = []
    for negative_sampler in negative_sampler_resolver.lookup_dict.keys():
        sampler = negative_sampler_resolver.make(
            query=negative_sampler,
            triples_factory=dataset.training,
            num_negs_per_pos=num_samples,
        )
        positive_batches = tqdm(
            dataset.training.mapped_triples.split(split_size=batch_size, dim=0),
            unit="batch",
            unit_scale=True,
            desc=f'FNR {sampler.get_normalized_name()}, {filterer_key}',
        )
        for positive_batch in positive_batches:
            positive_batches.set_postfix(batch_size=batch_size, dataset=dataset.get_normalized_name())
            negative_batch = sampler.corrupt_batch(positive_batch=positive_batch)
            false_negative_rates = filterer.contains(
                batch=negative_batch.view(-1, 3)
            ).view(negative_batch.shape[:-1]).float().mean(dim=-1)
            data.extend(
                (
                    USER,
                    pykeen.version.get_git_hash(),
                    dataset.get_normalized_name(),
                    negative_sampler,
                    filterer_key,
                    false_negative_rate,
                )
                for false_negative_rate in false_negative_rates.tolist()
            )
    df = pd.DataFrame(data=data, columns=["user", "hash", "dataset", "sampler", "filterer", "fnr"])
    df.to_csv(output_path, sep="\t", index=False)
    return df


def _plot_fnr(df: pd.DataFrame, *, directory: pathlib.Path, key: str):
    g = sns.catplot(
        data=df,
        x="sampler",
        y="fnr",
        col="dataset",
        kind="box",
        height=3.5,
        col_wrap=4 if df.dataset.nunique() > 4 else None,
    ).set_axis_labels(
        "", "False Negative Rate",
    )
    g.tight_layout()
    g.fig.suptitle(_prep_title('False Negative Rate Results'), fontsize=22, y=0.98)
    make_space_above(g.axes, topmargin=1.0)

    directory.mkdir(exist_ok=True, parents=True)
    figure_path_stem = directory.joinpath(key)
    # plt.savefig(figure_path_stem.with_suffix('.svg'))  # no SVG because too many outliers
    plt.savefig(figure_path_stem.with_suffix('.pdf'))
    plt.savefig(figure_path_stem.with_suffix('.png'), dpi=300)


def _prep_title(s: str) -> str:
    title = f'{s} from {USER}/{pykeen.version.get_version(with_git_hash=True)}'
    if (branch := pykeen.version.get_git_branch()) is not None:
        title += f' ({branch})'
    return title


def _prep_dir(
    directory: pathlib.Path,
    key: Optional[str] = None,
    hashed: bool = True,
    user_marked: bool = True,
) -> pathlib.Path:
    directory = directory.resolve()
    if user_marked:
        directory = directory.joinpath(USER)
    if hashed:
        directory = directory.joinpath(pykeen.version.get_git_hash())
    if key:
        directory = directory.joinpath(key)
    directory.mkdir(exist_ok=True, parents=True)
    return directory


def _iterate_datasets(dataset: Optional[str]) -> Iterable[Dataset]:
    if dataset:
        _dataset_list = [dataset]
    else:
        _dataset_list = _datasets
    it = tqdm(_dataset_list, desc='Dataset', disable=True)
    for dataset in it:
        it.set_postfix(dataset=dataset)
        yield get_dataset(dataset=dataset)


def make_space_above(axes, topmargin: float = 1.0) -> None:
    """Increase figure size to make topmargin (in inches) space for titles, without changing the axes sizes.

    :param axes: The array of axes.
    :param topmargin: The margin (in inches) to impose on the top of the figure.

    .. seealso:: Credit to https://stackoverflow.com/a/55768955/5775947
    """
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    width, height = fig.get_size_inches()

    fig_height = height - (1.0 - s.top) * height + topmargin
    fig.subplots_adjust(
        bottom=s.bottom * height / fig_height, top=1 - topmargin / fig_height
    )
    fig.set_figheight(fig_height)


if __name__ == '__main__':
    benchmark()
