# Cross-Market Transfer Learning for Stock Return Prediction

Rutgers Business School · Machine Learning in Finance · Professor Serdar Dinc

Can a model trained on U.S. stocks improve next-month return predictions in the UK and Japan when local training histories are short?

**Open `Cross_Market_Transfer_Learning.ipynb`.** It contains the complete data acquisition, preprocessing, models, experiments, evaluation, and interpretation. There is one research notebook and no separate model modules.

## Setup

Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name cross-market-wrds --display-name "Python (Cross-Market WRDS)"
jupyter lab Cross_Market_Transfer_Learning.ipynb
```

Select **Python (Cross-Market WRDS)**. On macOS, XGBoost requires `brew install libomp`; the notebook sets OpenMP limits before importing numerical libraries. GPU systems need a CUDA-compatible PyTorch installation.

## Run

1. Start with `RUN_MODE = "smoke"` in Part 1.
2. Run the notebook from the top. **Part 2.3** privately prompts for your WRDS username and password. Approve MFA if requested. Never enter credentials into saved source cells.
3. The first run downloads the required USA, UK, and Japan observations from WRDS into `data/raw/`. Completed country-year files are reused. An interrupted download repeats only missing partitions.
4. Later runs use local files with **no WRDS connection or download**. Restarting Jupyter does not delete the cache. You can rerun individual cells once their dependencies are defined.
5. Set `RUN_MODE = "full"` for complete cross-sections, tuning, three neural seeds, and final evaluation on suitable hardware.

The data snapshot does not refresh automatically. Keep `data/raw/` intact. To intentionally acquire a new release, use a new `RAW_DIR`.

**Offline / Amarel:** copy the complete `data/raw/` directory, including its manifest, and set `DATA_METHOD = "local"`. Alternatively, Part 2.2 imports WRDS country exports. Activate the environment and supply your allocation's partition/account/GPU options when submitting `amarel_job.sbatch`. Do not run full training on a login node.

## Research design

- Official JKP characteristics and stock screens; training-only feature selection; country-month ranks; no imputed future returns.
- Splits use the predicted return month: U.S. training 2000–2017, local histories of 1/3/5 years ending in 2017, validation 2018–2019, testing from 2020.
- Local Elastic Net and XGBoost; cross-sectional Transformer with local, zero-shot, fine-tuned, and pooled training; fixed equal-average ensemble.
- Equal-month neural losses, early stopping, resumable checkpoints, bounded tuning grids. Winning first-seed checkpoints and history-independent zero-shot forecasts are reused.
- Predictive metrics, decile portfolios, turnover, 0/10/25 bps cost sensitivities, paired six-month block bootstrap, and global Holm correction.

Results are saved automatically under `results/<mode>/<run_id>/`, with models under `models/<mode>/<run_id>/`. The run ID records code, data, predictors, settings, and software versions. Raw data remain shared across run modes. Tables 1–8 and figures are generated from actual execution.

## Files and validation

```text
Cross_Market_Transfer_Learning.ipynb   Complete research workflow
README.md                            Setup and usage
requirements.txt                     Compatible Python dependencies
amarel_job.sbatch                     Optional cluster execution
tests/test_notebook.py                Scientific and cache-reuse checks
.gitignore                           Excludes credentials and generated data
```

```bash
python -m pytest -q -p no:cacheprovider tests/test_notebook.py
```

Tests use temporary synthetic fixtures, including two complete notebook executions. They check chronology, feature leakage, ranks, attention equivariance, gradients, checkpoint resumption, metrics, portfolios, bootstrap pairing, and cache reuse with network/login calls forbidden. Fixtures and their generated results stay outside the project.

**Execution status:** the pipeline is implemented and tested with synthetic fixtures. Real WRDS acquisition, full-scale CUDA/Amarel execution, and empirical findings remain pending. Smoke results are software diagnostics, not research conclusions. Missing selected portfolio returns invalidate that portfolio-month; cost sensitivities are assumptions, and backtests omit borrow constraints and market impact.

Data, credentials, model binaries, executed notebooks, and generated results are excluded from Git. The notebook creates its runtime directories when needed.

Sources: [JKP WRDS guide](https://jkpfactors.com/jkp-wrds-guide), [official factor metadata](https://github.com/bkelly-lab/jkp-data/blob/main/src/jkp/data/resources/factor_details.xlsx), [JKP documentation](https://jkpfactors-data.s3.amazonaws.com/documents/Documentation.pdf).
