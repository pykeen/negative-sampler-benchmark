# negative-sampler-benchmark

🪑 Benchmark PyKEEN's negative samplers' false negative rates

## Speed Performance

Run the speed performance benchmarks with:

```shell
$ python main.py times
```

![Times](img/cthoyt/2207eaef/times.svg)

## False Negative Rate

Run false negative rate benchmarks with:

```shell
$ python main.py fnr
```

The plot demonstrates that the pseudo-typed negative sampler should be used in combination with filtering. The Nations
dataset has a unique pattern because it was constructed under the closed world assumption (unlike most knowledge graphs)
and therefore most negative samples are false negatives.

![False Negative Rate](img/cthoyt/2207eaef/fnr.png)
