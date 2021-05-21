"""Main script."""

import logging
import pathlib
import sys

import click
import math
import more_click
import pandas
import pandas as pd
import seaborn as seaborn
import torch
from matplotlib import pyplot as plt
from more_click import force_option
from torch.utils.benchmark import Timer
from tqdm.auto import tqdm

from pykeen.datasets import datasets, get_dataset
from pykeen.sampling import negative_sampler_resolver
from pykeen.sampling.filtering import PythonSetFilterer, filterer_resolver

HERE = pathlib.Path(__file__).resolve().parent
measurement_root = HERE.joinpath("data")
plot_root = HERE.joinpath("img")
default_time_plot_path = plot_root.joinpath("times.pdf")
default_fnr_plot_path = plot_root.joinpath("fnr.pdf")

logger = logging.getLogger(__name__)


@click.group()
def main():
    """The main entry point."""


@main.group()
def benchmark():
    """Run benchmarking."""


@benchmark.command()
@click.option("-d", "--dataset", type=click.Choice(datasets, case_sensitive=False), default="nations")
@click.option("-b", "--num-random-batches", type=int, default=20)
@click.option("-o", "--directory", type=pathlib.Path, default=measurement_root)
@force_option
@more_click.verbose_option
def time(
    dataset: str,
    num_random_batches: int,
    directory: pathlib.Path,
    force: bool
):
    """Benchmark sampling time."""
    dataset_instance = get_dataset(dataset=dataset)

    directory = directory.resolve().joinpath('times')
    directory.mkdir(exist_ok=True, parents=True)
    output_path = directory.joinpath(dataset_instance.get_normalized_name()).with_suffix('.tsv')
    if output_path.exists() and not force:
        click.secho(f'Already calculated time results for {dataset_instance.get_normalized_name()}')
        sys.exit(0)

    batch_sizes = [2 ** i for i in range(1, min(16, int(math.log2(dataset_instance.training.num_triples))))]
    logger.info(f"Evaluating batch sizes {batch_sizes} for dataset {dataset}")

    data = []
    for negative_sampler in negative_sampler_resolver.lookup_dict.keys():
        sampler = negative_sampler_resolver.make(query=negative_sampler, triples_factory=dataset_instance.training)
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
            positive_batch_idx = torch.randperm(dataset_instance.training.num_triples)[:batch_size]
            positive_batch = dataset_instance.training.mapped_triples[positive_batch_idx]
            timer = Timer(
                stmt="sampler.corrupt_batch(positive_batch=positive_batch)",
                globals=dict(
                    sampler=sampler,
                    positive_batch=positive_batch,
                ),
            )
            measurement = timer.blocked_autorange()
            data.extend(
                (batch_size, batch_id, t, negative_sampler, dataset)
                for t in measurement.raw_times
            )
    df = pandas.DataFrame(data=data, columns=["batch_size", "batch_id", "time", "sampler", "dataset"])
    df.to_csv(output_path, sep="\t", index=False)


@benchmark.command()
@click.option("-d", "--dataset", type=click.Choice(datasets, case_sensitive=False), default="nations")
@click.option("-b", "--batch_size", type=int, default=32)
@click.option("-n", "--num-samples", type=int, default=4096)  # TODO: Determine this based on dataset?
@click.option("-o", "--directory", type=pathlib.Path, default=measurement_root)
@more_click.verbose_option
@force_option
def fnr(
    dataset: str,
    batch_size: int,
    num_samples: int,
    directory: pathlib.Path,
    force: bool,
):
    """Estimate false negative rate."""
    dataset_instance = get_dataset(dataset=dataset)

    directory = directory.resolve().joinpath('fnr')
    directory.mkdir(exist_ok=True, parents=True)
    output_path = directory.joinpath(dataset_instance.get_normalized_name()).with_suffix('.tsv')
    if output_path.exists() and not force:
        click.secho(f'Already calculated FNR results for {dataset_instance.get_normalized_name()}')
        sys.exit(0)

    # create index structure for existence check
    filterer = PythonSetFilterer(
        mapped_triples=torch.cat(
            [
                dataset_instance.training.mapped_triples,
                dataset_instance.validation.mapped_triples,
                # dataset_instance.testing, # TODO: should this be used?
            ],
            dim=0,
        ),
    )
    filterer_key= filterer_resolver.normalize_inst(filterer)
    data = []
    for negative_sampler in negative_sampler_resolver.lookup_dict.keys():
        sampler = negative_sampler_resolver.make(
            query=negative_sampler,
            triples_factory=dataset_instance.training,
            num_negs_per_pos=num_samples,
        )
        for positive_batch in tqdm(
            dataset_instance.training.mapped_triples.split(split_size=batch_size, dim=0),
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
                    dataset,
                    negative_sampler,
                    filterer_key,
                    false_negative_rate,
                )
                for false_negative_rate in false_negative_rates.tolist()
            )
    df = pandas.DataFrame(data=data, columns=["dataset", "sampler", "filterer", "fnr"])
    df.to_csv(output_path, sep="\t", index=False)


@main.group()
def plot():
    """Plot commands."""


@plot.command(name='times')
@click.option("-i", "--directory", type=pathlib.Path, default=measurement_root)
@click.option("-o", "--output", type=pathlib.Path, default=plot_root)
@more_click.verbose_option
def plot_times(
    directory: pathlib.Path,
    output: pathlib.Path,
):
    """Create plots."""
    key = 'times'
    df = pd.concat([
        pd.read_csv(path, sep='\t')
        for path in directory.joinpath(key).iterdir()
    ])
    g = seaborn.relplot(
        data=df,
        x="batch_size",
        y="time",
        hue="sampler",
        kind="line",
        col="dataset",
        # ci=100,
        # estimator=numpy.median,
    ).set(
        xscale="log",
        xlabel="batch size",
        ylabel="time (s)/batch",
    )
    g.tight_layout()

    output.mkdir(exist_ok=True, parents=True)
    figure_path_stem = output.joinpath(key)
    plt.savefig(figure_path_stem.with_suffix('.svg'))
    plt.savefig(figure_path_stem.with_suffix('.pdf'))
    plt.savefig(figure_path_stem.with_suffix('.png'), dpi=300)


@plot.command(name='fnr')
@click.option("-i", "--directory", type=pathlib.Path, default=measurement_root)
@click.option("-o", "--output", type=pathlib.Path, default=plot_root)
@more_click.verbose_option
def plot_fnr(
    directory: pathlib.Path,
    output: pathlib.Path,
):
    """Create false negative rate plots."""
    key = 'fnr'
    df = pd.concat([
        pd.read_csv(path, sep='\t')
        for path in directory.joinpath(key).iterdir()
    ])
    g = seaborn.catplot(
        data=df,
        x="sampler",
        y="fnr",
        col="dataset",
        kind="box",
    ).set(
        ylabel="False Negative Rate",
    )
    g.tight_layout()

    output.mkdir(exist_ok=True, parents=True)
    figure_path_stem = output.joinpath(key)
    plt.savefig(figure_path_stem.with_suffix('.svg'))
    plt.savefig(figure_path_stem.with_suffix('.pdf'))
    plt.savefig(figure_path_stem.with_suffix('.png'), dpi=300)


if __name__ == '__main__':
    main()
