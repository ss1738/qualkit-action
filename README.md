# qualkit-action

A GitHub Action that gates model pull requests on measured statistical
guarantees. When a PR changes a model (retraining, quantization, pruning), the
gate runs qualkit's pre-flight checks and **fails the build** if:

- the forecast conformal coverage under stale calibration falls below your
  floor (compression broke the guarantee: recalibrate or reduce compression), or
- the recalibrated prediction sets balloon past your size budget (the model is
  valid but uninformative at this precision).

As far as we can tell this is the first CI gate built on distribution-free
statistical guarantees rather than heuristic thresholds. The checks implement
measured findings from a 12-finding research programme:
github.com/ss1738/euroSAT-CNN-landcover (the coverage forecast used by the
gate held in 60/60 validation runs across 2 datasets, 2 architectures and 5
bit-widths).

## Use

Add a config module to your repo (see `example/qualkit_config.py`) defining
`get_reference_model()`, `get_deploy_model()`, `get_cal_loader()` and
optionally `get_probe_loader()`. Then:

```yaml
name: model-assurance
on: [pull_request]
jobs:
  qualkit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - uses: ss1738/qualkit-action@main
        with:
          config: qualkit_config.py
          alpha: "0.10"
          min-coverage-floor: "0.85"
          max-mean-setsize: "1.5"
```

The full assurance report (JSON + markdown) is written to the workspace and
summarized on the workflow run page.

## Honest scope

Classification with split conformal prediction, PyTorch. The coverage
forecast errs safe (never underestimates the loss; near-exact on peaked score
distributions). The SEU probe is a software bit-flip characterization, not a
radiation test.

MIT licence.
