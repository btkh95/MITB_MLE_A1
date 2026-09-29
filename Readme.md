# CS611 Assignment 1: Data Processing Pipelines

This project prepares customer and loan data for predicting loan default at the point of application. It uses PySpark and a medallion architecture (bronze, silver and gold) to produce cleaned features and default labels. Docker provides the Python, Java and JupyterLab environment.

The current implementation builds the datamart and prepares train, test and out-of-time (OOT) datasets. It does not yet train or evaluate a predictive model.

## How to run

### Prerequisites

- Git to clone the repository.
- Docker with Docker Compose available, with the Docker engine running.
- Port `8888` available for JupyterLab.
- The four source CSV files in `data/` (listed below).

### Build the datamart

1. Clone the repository and enter its root directory:

   ```sh
   git clone https://github.com/btkh95/MITB_MLE_A1.git
   cd MITB_MLE_A1
   ```

2. Build the image, then start the JupyterLab service:

   ```sh
   docker compose build
   docker compose up
   ```

   For subsequent runs, use only `docker compose up`. Run `docker compose build` again after changing `Dockerfile` or `requirements.txt`; changes to Python files do not require rebuilding.

3. Open [JupyterLab](http://localhost:8888/). From the Launcher, open **Terminal** and run:

   ```sh
   cd /app
   python main.py
   ```

   Alternatively, run the pipeline from a terminal on your host, in the repository root:

   ```sh
   docker compose exec jupyter python main.py
   ```

4. Wait for processing to finish. The script prints row counts and saved paths, then displays the combined gold labels. Outputs are written to the `datamart/` directory.

The repository is mounted at `/app` in the container, so generated files also appear in your local checkout. Starting JupyterLab does not automatically run the pipeline.

### Prepare model datasets

After `main.py` finishes successfully, run this in the JupyterLab terminal:

```sh
python model_train.py
```

Or run it from the host terminal:

```sh
docker compose exec jupyter python model_train.py
```

Despite its name, `model_train.py` currently performs dataset splitting and preprocessing only. It:

- Joins labels to customer features at the loan start date.
- Reserves the newest 20% of distinct loan application months for OOT validation.
- Splits earlier loans into 80% training and 20% test data, stratified by label, with random seed `55`.
- Learns numeric imputation values and scaling from training data, and aligns test/OOT categorical encodings to the training columns.
- Writes `train.parquet`, `test.parquet` and `oot.parquet` under `datamart/model_data/`, plus `model_bank/preprocessor.pkl`.

The saved datasets retain IDs, dates and labels for inspection; these are excluded from the predictor columns recorded in the preprocessor.

### Stop the environment

Run from the host terminal in the repository root:

```sh
docker compose down
```

Generated files remain in the local checkout. To inspect service logs, use `docker compose logs jupyter`.

## Pipeline behaviour

`main.py` processes first-of-month snapshots from **1 January 2023 to 1 December 2024**, inclusive. Change `start_date_str` and `end_date_str` in that file to adjust the backfill range. Bronze ingestion selects records whose snapshot date equals each processing date; it does not ingest every day within the month.

| Layer | Processing | Output format |
| --- | --- | --- |
| Bronze | Reads configured source CSVs, removes stray `x` characters from snapshot dates and selects each snapshot. | CSV files |
| Silver | Applies data types and cleaning rules, removes configured PII from attributes, flags invalid values, deduplicates clickstream rows and derives loan delinquency fields. | Parquet datasets |
| Gold | Creates default labels, engineers customer and financial features, aggregates earlier clickstream observations and joins the feature tables. | Parquet datasets |

The default label is `30dpd_6mob`: a loan receives label `1` when its derived days past due is at least 30 at month on book 6; otherwise it receives `0`. Only records at month on book 6 enter the label store. These thresholds are passed as `dpd=30` and `mob=6` in `main.py`.

Gold features are keyed by `Customer_ID` and `snapshot_date`. Attributes and financials are joined with a full outer join, then historical clickstream features are left joined. Clickstream aggregates use only processed observations strictly before the feature snapshot date. Null or duplicate feature join keys cause an error.

The pipeline regenerates `static/loan_types.yaml` from the cleaned silver financial data before building gold features. Rerunning overwrites outputs for the dates processed and the joined feature store; it does not remove older partitions outside the configured range. Downstream wildcard reads can include those older partitions.

## Generated outputs

```text
datamart/
  bronze/
    lms/
    fe_attr/
    fe_fin/
    fe_click/
  silver/
    loan_daily/
    fe_attr/
    fe_fin/
    fe_click/
  gold/
    label_store/
    feature_store/
      fe_attr/
      fe_fin/
      fe_click/
      gold_joined_features.parquet/
  model_data/                     # Created by model_train.py
    train.parquet/
    test.parquet/
    oot.parquet/
model_bank/                       # Created by model_train.py
  preprocessor.pkl
```

Spark writes each `.parquet` output as a directory containing part files, rather than a single file. Generated datamart and model-bank outputs are ignored by Git.

## Files in the repository

| Path | Purpose |
| --- | --- |
| `data/` | Source CSV datasets. |
| `main.py` | Runs the bronze, silver and gold backfill and assembles the feature store. |
| `model_train.py` | Prepares train/test/OOT datasets and saves fitted preprocessing. |
| `utils/` | Layer transformations, source-file checks and YAML helpers. |
| `static/` | Source filenames, data types, PII columns, valid ranges and loan-type categories. |
| `EDA FIles/` | Exploratory notebooks for source, silver and gold data. |
| `data_issues_found.md` | Observed data-quality issues and cleaning decisions. |
| `Dockerfile` | Container environment based on Python 3.12, with Java and Python dependencies. |
| `docker-compose.yaml` | JupyterLab service, port mapping and repository mount. |
| `requirements.txt` | Python dependencies, including PySpark 3.5.5. |
| `Assignment 1.pdf` | Assignment brief. |

## Raw datasets

Source filenames are configured in `static/dir.yaml` and resolved under `data/`.

| File | Contents |
| --- | --- |
| `feature_clickstream.csv` | Customer clickstream observations with 20 anonymised features (`fe_1` to `fe_20`) and a snapshot date. |
| `features_attributes.csv` | Customer attributes, including name, age, SSN and occupation. |
| `features_financials.csv` | Customer financial information, including income, credit accounts, loans and monthly investments. |
| `lms_loan_daily.csv` | Loan snapshots, including start date, instalment number, amounts due and paid, overdue amount and balance. |

At startup, the pipeline reports missing or unexpected CSV files. Missing inputs must be supplied before processing can complete.
