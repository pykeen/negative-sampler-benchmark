"""Main script."""

import logging
import pathlib
from typing import Iterable, Optional

import click
import math
import pandas
import pandas as pd
import seaborn as seaborn
import torch
from docdata import get_docdata
from matplotlib import pyplot as plt
from more_click import force_option, verbose_option
from torch.utils.benchmark import Timer
from tqdm.auto import tqdm

from pykeen.datasets import Dataset, datasets as datasets_dict, get_dataset
from pykeen.sampling import negative_sampler_resolver
from pykeen.sampling.filtering import PythonSetFilterer, filterer_resolver

HERE = pathlib.Path(__file__).resolve().parent
measurement_root = HERE.joinpath("data")
plot_root = HERE.joinpath("img")
default_time_plot_path = plot_root.joinpath("times.pdf")
default_fnr_plot_path = plot_root.joinpath("fnr.pdf")

logger = logging.getLogger(__name__)

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


@click.group()
def main():
    """The main entry point."""


@main.group()
def benchmark():
    """Run benchmarking."""


@benchmark.command()
@click.option("-d", "--dataset", type=click.Choice(datasets_dict, case_sensitive=False))
@click.option("-b", "--num-random-batches", type=int, default=20)
@click.option("-o", "--directory", type=pathlib.Path, default=measurement_root)
@click.option("--plot-directory", type=pathlib.Path, default=plot_root)
@force_option
@verbose_option
def time(
    dataset: str,
    num_random_batches: int,
    directory: pathlib.Path,
    plot_directory: pathlib.Path,
    force: bool
) -> None:
    """Benchmark sampling time."""
    directory = directory.resolve().joinpath('times')
    directory.mkdir(exist_ok=True, parents=True)
    dfs = []
    for dataset_instance in _iterate_datasets(dataset):
        dfs.append(_time_helper(
            dataset_instance,
            directory=directory,
            num_random_batches=num_random_batches,
            force=force,
        ))
    _plot_times(pd.concat(dfs), directory=plot_directory)


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
            progress.set_postfix(batch_size=batch_size)
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
                (batch_size, batch_id, t, negative_sampler, dataset.get_normalized_name())
                for t in measurement.raw_times
            )
    df = pandas.DataFrame(data=data, columns=["batch_size", "batch_id", "time", "sampler", "dataset"])
    df.to_csv(output_path, sep="\t", index=False)
    return df


def _plot_times(df: pd.DataFrame, directory: pathlib.Path = plot_root, key: str = 'times'):
    g = seaborn.relplot(
        data=df,
        x="batch_size",
        y="time",
        hue="sampler",
        kind="line",
        col="dataset",
        col_wrap=4 if df.dataset.nunique() > 4 else None,
        # ci=100,
        # estimator=numpy.median,
    ).set(
        xscale="log",
        xlabel="batch size",
        ylabel="time (s)/batch",
    )
    g.tight_layout()

    directory.mkdir(exist_ok=True, parents=True)
    figure_path_stem = directory.joinpath(key)
    plt.savefig(figure_path_stem.with_suffix('.svg'))
    plt.savefig(figure_path_stem.with_suffix('.pdf'))
    plt.savefig(figure_path_stem.with_suffix('.png'), dpi=300)


@benchmark.command()
@click.option("-d", "--dataset", type=click.Choice(datasets_dict, case_sensitive=False))
@click.option("-b", "--batch_size", type=int, default=32)
@click.option("-n", "--num-samples", type=int, default=4096)  # TODO: Determine this based on dataset?
@click.option("-o", "--directory", type=pathlib.Path, default=measurement_root)
@click.option("--plot-directory", type=pathlib.Path, default=plot_root)
@verbose_option
@force_option
def fnr(
    dataset: str,
    batch_size: int,
    num_samples: int,
    directory: pathlib.Path,
    plot_directory: pathlib.Path,
    force: bool,
):
    """Estimate false negative rate."""
    directory = directory.resolve().joinpath('fnr')
    directory.mkdir(exist_ok=True, parents=True)
    dfs = []
    for dataset_instance in _iterate_datasets(dataset):
        dfs.append(_fnr_helper(
            dataset_instance,
            directory=directory,
            num_samples=num_samples,
            batch_size=batch_size,
            force=force,
        ))
    _plot_fnr(pd.concat(dfs), plot_directory)


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
        mapped_triples=torch.cat(
            [
                dataset.training.mapped_triples,
                dataset.validation.mapped_triples,
                # dataset_instance.testing, # TODO: should this be used?
            ],
            dim=0,
        ),
    )
    filterer_key = filterer_resolver.normalize_inst(filterer)
    data = []
    for negative_sampler in negative_sampler_resolver.lookup_dict.keys():
        sampler = negative_sampler_resolver.make(
            query=negative_sampler,
            triples_factory=dataset.training,
            num_negs_per_pos=num_samples,
        )
        for positive_batch in tqdm(
            dataset.training.mapped_triples.split(split_size=batch_size, dim=0),
            unit="batch",
            unit_scale=True,
            desc=f'FNR {sampler.get_normalized_name()}, {filterer_key}',
        ):
            negative_batch = sampler.corrupt_batch(positive_batch=positive_batch)
            false_negative_rates = filterer.contains(
                batch=negative_batch.view(-1, 3)
            ).view(negative_batch.shape[:-1]).float().mean(dim=-1)
            data.extend(
                (
                    dataset.get_normalized_name(),
                    negative_sampler,
                    filterer_key,
                    false_negative_rate,
                )
                for false_negative_rate in false_negative_rates.tolist()
            )
    df = pandas.DataFrame(data=data, columns=["dataset", "sampler", "filterer", "fnr"])
    df.to_csv(output_path, sep="\t", index=False)
    return df


def _plot_fnr(df: pd.DataFrame, directory: pathlib.Path, key: str = 'fnr'):
    g = seaborn.catplot(
        data=df,
        x="sampler",
        y="fnr",
        col="dataset",
        kind="box",
        col_wrap=4 if df.dataset.nunique() > 4 else None,
    ).set_axis_labels(
        "", "False Negative Rate",
    )
    g.tight_layout()

    directory.mkdir(exist_ok=True, parents=True)
    figure_path_stem = directory.joinpath(key)
    plt.savefig(figure_path_stem.with_suffix('.svg'))
    plt.savefig(figure_path_stem.with_suffix('.pdf'))
    plt.savefig(figure_path_stem.with_suffix('.png'), dpi=300)


def _iterate_datasets(dataset: Optional[str]) -> Iterable[Dataset]:
    if dataset:
        _dataset_list = [dataset]
    else:
        _dataset_list = _datasets
    it = tqdm(_dataset_list, desc='Dataset')
    for dataset in it:
        it.set_postfix(dataset=dataset)
        yield get_dataset(dataset=dataset)


if __name__ == '__main__':
    main()
