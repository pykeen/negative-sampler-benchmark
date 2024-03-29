# -*- coding: utf-8 -*-

"""Main script."""

import getpass
import logging
import math
import pathlib
from itertools import product
from textwrap import dedent
from typing import Iterable, Optional, cast

import click
import pandas as pd
import pykeen.version
import seaborn as sns
import torch
from docdata import get_docdata
from matplotlib import pyplot as plt
from more_click import force_option, verbose_option
from pykeen.datasets import dataset_resolver, get_dataset
from pykeen.datasets.base import Dataset
from pykeen.sampling import negative_sampler_resolver
from pykeen.sampling.filtering import PythonSetFilterer
from torch.utils.benchmark import Timer
from tqdm import tqdm

logger = logging.getLogger(__name__)

USER = getpass.getuser()
VERSION = pykeen.get_version()
GIT_HASH = pykeen.get_git_hash()
GIT_BRANCH = pykeen.get_git_branch()

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_DIRECTORY = HERE.joinpath("data", USER, GIT_HASH)

TIMES_KEY = "times"
FNR_KEY = "fnr"
TIMES_COLUMNS = ["batch_size", "batch_id", "time"]
FNR_COLUMNS = ["fnr"]


@click.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(dataset_resolver.lookup_dict, case_sensitive=False),
)
@click.option("--num-random-batches", type=int, default=20)
@click.option("--batch-size", type=int, default=32)
@click.option(
    "-n", "--num-samples", type=int, default=4096
)  # TODO: Determine this based on dataset?
@click.option("-o", "--directory", type=pathlib.Path, default=DEFAULT_DIRECTORY)
@click.option("--skip-fnr", is_flag=True)
@force_option
@verbose_option
def benchmark(
    dataset: Optional[str],
    num_random_batches: int,
    batch_size: int,
    num_samples: int,
    directory: pathlib.Path,
    skip_fnr: bool,
    force: bool,
) -> None:
    """Benchmark negative sampling."""
    times_dfs, fnr_dfs = [], []
    for dataset_instance in _iterate_datasets(dataset):
        times_dfs.append(
            _time_helper(
                dataset_instance,
                directory=directory.joinpath(TIMES_KEY),
                num_random_batches=num_random_batches,
                force=force,
            )
        )
        if not skip_fnr:
            fnr_dfs.append(
                _fnr_helper(
                    dataset_instance,
                    directory=directory.joinpath(FNR_KEY),
                    num_samples=num_samples,
                    batch_size=batch_size,
                    force=force,
                )
            )

    sns.set_style("whitegrid")
    _plot_times(pd.concat(times_dfs), key=TIMES_KEY, directory=directory)
    if not skip_fnr:
        _plot_fnr(pd.concat(fnr_dfs), key=FNR_KEY, directory=directory)

    with directory.joinpath("README.md").open("w") as file:
        print(
            dedent(
                f"""\
        # Results on PyKEEN v{VERSION} ({GIT_BRANCH})

        - See commit: [{GIT_HASH}](https://github.com/pykeen/pykeen/commit/{GIT_HASH})
        - Browse commit: [{GIT_HASH}](https://github.com/pykeen/pykeen/tree/{GIT_HASH})

        Run again with:
        
        ```shell
        $ git clone https://github.com/pykeen/pykeen.git
        $ cd pykeen
        $ git checkout 2207eaef
        $ pip install -e .
        $ cd ..
        $ git clone https://github.com/pykeen/negative-sampler-benchmark.git
        $ cd negative-sampler-benchmark
        $ python main.py
        ```

        ## Speed Performance

        ![Times](times.svg)
        
        ## False Negative Rate

        ![False Negative Rate](fnr.png)        
        """
            ),
            file=file,
        )


def _time_helper(
    dataset: Dataset,
    *,
    directory: pathlib.Path,
    num_random_batches: int,
    force: bool = False,
) -> pd.DataFrame:
    dataset_directory = directory.joinpath(dataset.get_normalized_name())
    dataset_directory.mkdir(exist_ok=True, parents=True)

    batch_sizes = [
        2**i for i in range(1, min(16, int(math.log2(dataset.training.num_triples))))
    ]
    logger.info(
        f"Evaluating batch sizes {batch_sizes} for dataset {dataset.get_normalized_name()}"
    )

    dfs = []
    sampler_it = tqdm(list(negative_sampler_resolver), desc="Sampler", leave=False)
    for negative_sampler_cls in sampler_it:
        sampler_it.set_postfix(
            sampler=negative_sampler_cls.get_normalized_name(),
            dataset=dataset.get_normalized_name(),
        )
        path = dataset_directory.joinpath(
            negative_sampler_cls.get_normalized_name()
        ).with_suffix(".tsv.gz")
        if path.exists() and not force:
            df = pd.read_csv(path, sep="\t")
        else:
            negative_sampler = negative_sampler_resolver.make(
                query=negative_sampler_cls,
                triples_factory=dataset.training,
            )
            progress = tqdm(
                product(batch_sizes, range(num_random_batches)),
                unit_scale=True,
                desc="Batch",
                total=len(batch_sizes) * num_random_batches,
                leave=False,
            )
            data = []
            for batch_size, batch_id in progress:
                progress.set_postfix(
                    size=batch_size,
                    id=batch_id,
                    sampler=negative_sampler_cls.get_normalized_name(),
                    dataset=dataset.get_normalized_name(),
                )
                positive_batch_idx = torch.randperm(dataset.training.num_triples)[
                    :batch_size
                ]
                positive_batch = dataset.training.mapped_triples[positive_batch_idx]
                timer = Timer(
                    stmt="sampler.corrupt_batch(positive_batch=positive_batch)",
                    globals=dict(
                        sampler=negative_sampler,
                        positive_batch=positive_batch,
                    ),
                )
                measurement = timer.blocked_autorange()
                data.extend((batch_size, batch_id, t) for t in measurement.raw_times)
            df = pd.DataFrame(data=data, columns=TIMES_COLUMNS)
            df.to_csv(path, sep="\t", index=False)

        df["dataset"] = dataset.get_normalized_name()
        df["sampler"] = negative_sampler_cls.get_normalized_name()
        df = df[["dataset", "sampler", *TIMES_COLUMNS]]
        dfs.append(df)

    return pd.concat(dfs)


def _plot_times(df: pd.DataFrame, *, key: str, directory: pathlib.Path):
    g = sns.relplot(
        data=df,
        x="batch_size",
        y="time",
        hue="sampler",
        kind="line",
        col="dataset",
        height=3.5,
        col_wrap=4 if df.dataset.nunique() > 4 else None,
        ci="sd",
        estimator="mean",
    ).set(
        xscale="log",
        xlabel="Batch Size",
        ylabel="Seconds Per Batch",
    )
    g.tight_layout()
    g.fig.suptitle(_prep_title("Time Results"), fontsize=22, y=0.98)
    make_space_above(g.axes, topmargin=1.0)

    directory.mkdir(exist_ok=True, parents=True)
    figure_path_stem = directory.joinpath(key)
    plt.savefig(figure_path_stem.with_suffix(".svg"))
    plt.savefig(figure_path_stem.with_suffix(".pdf"))
    plt.savefig(figure_path_stem.with_suffix(".png"), dpi=300)


def _fnr_helper(
    dataset: Dataset,
    *,
    directory: pathlib.Path,
    num_samples: int,
    batch_size: int,
    force: bool = False,
) -> pd.DataFrame:
    dataset_directory = directory.joinpath(dataset.get_normalized_name())
    dataset_directory.mkdir(exist_ok=True, parents=True)

    # create index structure for existence check
    filterer = PythonSetFilterer(
        mapped_triples=cast(
            torch.LongTensor,
            torch.cat(
                [
                    dataset.training.mapped_triples,
                    dataset.validation.mapped_triples,
                    # dataset_instance.testing, # TODO: should this be used?
                ],
                dim=0,
            ),
        ),
    )
    dfs = []
    sampler_it = tqdm(list(negative_sampler_resolver), desc="Sampler", leave=False)
    for negative_sampler_cls in sampler_it:
        sampler_it.set_postfix(
            sampler=negative_sampler_cls.get_normalized_name(),
            dataset=dataset.get_normalized_name(),
        )
        path = dataset_directory.joinpath(
            negative_sampler_cls.get_normalized_name()
        ).with_suffix(".tsv.gz")
        if path.exists() and not force:
            df = pd.read_csv(path, sep="\t")
        else:
            sampler = negative_sampler_resolver.make(
                query=negative_sampler_cls,
                triples_factory=dataset.training,
                num_negs_per_pos=num_samples,
            )
            positive_batches = tqdm(
                dataset.training.mapped_triples.split(split_size=batch_size, dim=0),
                unit="batch",
                unit_scale=True,
                desc="Batch",
            )
            data = []
            for positive_batch in positive_batches:
                positive_batches.set_postfix(
                    batch_size=batch_size,
                    sampler=negative_sampler_cls.get_normalized_name(),
                    dataset=dataset.get_normalized_name(),
                )
                negative_batch = sampler.corrupt_batch(positive_batch=positive_batch)
                false_negative_rates = (
                    filterer.contains(batch=negative_batch.view(-1, 3))
                    .view(negative_batch.shape[:-1])
                    .float()
                    .mean(dim=-1)
                )
                data.extend(false_negative_rates.tolist())
            df = pd.DataFrame(data, columns=FNR_COLUMNS)
            df.to_csv(path, sep="\t", index=False)

        df["dataset"] = dataset.get_normalized_name()
        df["sampler"] = negative_sampler_cls.get_normalized_name()
        df = df[["dataset", "sampler", *FNR_COLUMNS]]
        dfs.append(df)

    return pd.concat(dfs)


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
        "",
        "False Negative Rate",
    )
    g.tight_layout()
    g.fig.suptitle(_prep_title("False Negative Rate Results"), fontsize=22, y=0.98)
    make_space_above(g.axes, topmargin=1.0)

    directory.mkdir(exist_ok=True, parents=True)
    figure_path_stem = directory.joinpath(key)
    # plt.savefig(figure_path_stem.with_suffix('.svg'))  # no SVG because too many outliers
    plt.savefig(figure_path_stem.with_suffix(".pdf"))
    plt.savefig(figure_path_stem.with_suffix(".png"), dpi=300)


def _prep_title(s: str) -> str:
    title = f"{s} from {USER}/{pykeen.version.get_version(with_git_hash=True)}"
    if (branch := pykeen.version.get_git_branch()) is not None:
        title += f" ({branch})"
    return title


def _iterate_datasets(dataset: Optional[str]) -> Iterable[Dataset]:
    if dataset:
        _dataset_list = [dataset]
    else:
        _dataset_list = _get_datasets()
    it = tqdm(_dataset_list, desc="Dataset")
    for dataset in it:
        it.set_postfix(dataset=dataset)
        yield get_dataset(dataset=dataset)


def _triples(d: str) -> int:
    return get_docdata(dataset_resolver.lookup_dict[d])["statistics"]["triples"]


def _get_datasets():
    rv = sorted(dataset_resolver.lookup_dict, key=_triples)
    return rv[: rv.index("fb15k237") + 1]  # include fb15k-237


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


if __name__ == "__main__":
    benchmark()
