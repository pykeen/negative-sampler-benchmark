"""Main script."""
import logging
import math
import pathlib

import click
import more_click
import pandas
import seaborn as seaborn
import torch
from matplotlib import pyplot as plt
from pykeen.datasets import datasets, get_dataset
from pykeen.sampling import negative_sampler_resolver
from pykeen.sampling.filtering import PythonSetFilterer
from torch.utils.benchmark import Timer
from tqdm.auto import tqdm

HERE = pathlib.Path(__file__).resolve().parent
measurement_root = HERE.joinpath("data")
plot_root = HERE.joinpath("img")
default_times_path = measurement_root.joinpath("times.tsv")
default_fnr_path = measurement_root.joinpath("fnr.tsv")
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
@click.option("-o", "--output-path", type=pathlib.Path, default=default_times_path)
@more_click.verbose_option
def time(
    dataset: str,
    num_random_batches: int,
    output_path: pathlib.Path,
):
    """Benchmark sampling time."""
    dataset_instance = get_dataset(dataset=dataset)
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
                )
            )
            measurement = timer.blocked_autorange()
            data.extend(
                (batch_size, batch_id, t, negative_sampler, dataset)
                for t in measurement.raw_times
            )
    df = pandas.DataFrame(data=data, columns=["batch_size", "batch_id", "time", "sampler", "dataset"])
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_path, sep="\t", index=False)


@benchmark.command()
@click.option("-d", "--dataset", type=click.Choice(datasets, case_sensitive=False), default="nations")
@click.option("-b", "--batch_size", type=int, default=32)
@click.option("-n", "--num-samples", type=int, default=4096)  # TODO: Determine this based on dataset?
@click.option("-o", "--output-path", type=pathlib.Path, default=default_fnr_path)
@more_click.verbose_option
def fnr(
    dataset: str,
    batch_size: int,
    num_samples: int,
    output_path: pathlib.Path,
):
    """Estimate false negative rate."""
    dataset_instance = get_dataset(dataset=dataset)
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
        ):
            negative_batch = sampler.corrupt_batch(positive_batch=positive_batch)
            false_negative_rates = filterer.contains(
                batch=negative_batch.view(-1, 3)
            ).view(negative_batch.shape[:-1]).float().mean(dim=-1)
            data.extend(
                (
                    dataset,
                    negative_sampler,
                    false_negative_rate,
                )
                for false_negative_rate in false_negative_rates.tolist()
            )
    df = pandas.DataFrame(data=data, columns=["dataset", "sampler", "fnr"])
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_path, sep="\t", index=False)


@main.command()
@click.option("-i", "--input-path", type=pathlib.Path, default=default_times_path)
@click.option("-o", "--output-path", type=pathlib.Path, default=default_time_plot_path)
@more_click.verbose_option
def plot(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
):
    """Create plots."""
    df = pandas.read_csv(input_path, sep="\t")
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
        ylabel="s/batch",
    )
    g.tight_layout()
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path)


@main.command()
@click.option("-i", "--input-path", type=pathlib.Path, default=default_fnr_path)
@click.option("-o", "--output-path", type=pathlib.Path, default=default_fnr_plot_path)
@more_click.verbose_option
def plot2(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
):
    """Create false negative rate plots."""
    df = pandas.read_csv(input_path, sep="\t")
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
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path)


if __name__ == '__main__':
    main()
