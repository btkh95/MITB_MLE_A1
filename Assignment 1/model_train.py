"""Model sanity check: make train, test and out-of-time datasets from the gold stores.

Run after main.py has built the datamart:  python model_train.py
"""

import math
import os
import pickle

import pandas as pd
import pyspark
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def process_model_splits(feature_directory, label_directory, output_directory,
                         model_bank_directory, spark, oot_fraction=0.2,
                         test_size=0.2, random_state=55):
    # 1. Join labels to features from the date the loan started.
    # The label date is later, so keep it separate from the feature date.
    labels = spark.read.parquet(label_directory + "gold_label_store_*.parquet")
    features = spark.read.parquet(feature_directory + "gold_joined_features.parquet")

    labels = labels.withColumnRenamed("snapshot_date", "label_snapshot_date")
    features = features.withColumnRenamed("snapshot_date", "feature_snapshot_date")
    data = labels.withColumn("feature_snapshot_date", F.col("loan_start_date"))
    data = data.join(features, on=["Customer_ID", "feature_snapshot_date"], how="inner")
    print("labeled loans with application-date features:", data.count())
    data = data.toPandas()

    # 2. Set aside the newest 20% of application months as OOT.
    # Split the earlier loans 80/20 for train and test, keeping default rates similar.
    data["loan_start_date"] = pd.to_datetime(data["loan_start_date"])
    months = sorted(data["loan_start_date"].unique())
    oot_cutoff = months[-math.ceil(len(months) * oot_fraction)]
    earlier = data[data.loan_start_date < oot_cutoff]
    oot = data[data.loan_start_date >= oot_cutoff]
    train, test = train_test_split(earlier, test_size=test_size,
                                   random_state=random_state, stratify=earlier.label)

    # 3. IDs, dates and labels are for joins/audits, not model predictors.
    metadata_columns = ["loan_id", "Customer_ID", "loan_start_date",
                        "feature_snapshot_date", "label_snapshot_date", "label_def", "label"]
    feature_columns = [c for c in data.columns if c not in metadata_columns]
    X_train = train[feature_columns].copy()
    X_test = test[feature_columns].copy()
    X_oot = oot[feature_columns].copy()
    numeric_columns = [c for c in feature_columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical_columns = [c for c in feature_columns if c not in numeric_columns]

    # 4. Using TRAIN only: mean for age/counts, mean among clickers for fe_*_mean, median otherwise.
    mean_columns = ["Age", "Num_Bank_Accounts", "Num_Credit_Card",
                    "Num_of_Loan", "Num_of_Delayed_Payment"]
    click_columns = [c for c in numeric_columns if c.startswith("fe_") and c.endswith("_mean")]
    fill_values = {}
    for column in numeric_columns:
        if column in mean_columns:
            fill_values[column] = X_train[column].mean()
        elif column in click_columns:
            fill_values[column] = X_train.loc[train["has_clickstream"] == 1, column].mean()
        else:
            fill_values[column] = X_train[column].median()
    for X in [X_train, X_test, X_oot]:
        X[numeric_columns] = X[numeric_columns].fillna(fill_values)

    # 5. Turn categories into 0/1 columns. Test and OOT use the same columns
    # as training; a category seen only later does not add a new column.
    X_train = pd.get_dummies(X_train, columns=categorical_columns, dtype=int)
    X_test = pd.get_dummies(X_test, columns=categorical_columns, dtype=int)
    X_oot = pd.get_dummies(X_oot, columns=categorical_columns, dtype=int)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_oot = X_oot.reindex(columns=X_train.columns, fill_value=0)

    # 6. Fit StandardScaler() on TRAIN numeric values, then reuse it for
    # test and OOT. Keep the True/False flag columns unchanged.
    scale_columns = [c for c in numeric_columns if not pd.api.types.is_bool_dtype(train[c])]
    scaler = StandardScaler()
    X_train[scale_columns] = scaler.fit_transform(X_train[scale_columns])
    X_test[scale_columns] = scaler.transform(X_test[scale_columns])
    X_oot[scale_columns] = scaler.transform(X_oot[scale_columns])

    # 7. Save the fitted preprocessing so new loans can be prepared the same way
    # at prediction time.
    os.makedirs(model_bank_directory, exist_ok=True)
    preprocessor = {
        "fill_values": fill_values,
        "categorical_columns": categorical_columns,
        "model_columns": list(X_train.columns),
        "scale_columns": scale_columns,
        "scaler": scaler,
    }
    filepath = model_bank_directory + "preprocessor.pkl"
    with open(filepath, "wb") as file:
        pickle.dump(preprocessor, file)
    print("saved to:", filepath)

    # 8. Save three datasets. Loan IDs, dates and labels stay in the files
    # for inspection; they are not model predictors.
    os.makedirs(output_directory, exist_ok=True)
    for name, rows, X in [("train", train, X_train),
                          ("test", test, X_test), ("oot", oot, X_oot)]:
        metadata = rows[metadata_columns].copy()
        for column in ["loan_start_date", "feature_snapshot_date", "label_snapshot_date"]:
            metadata[column] = pd.to_datetime(metadata[column]).dt.date
        output = pd.concat([metadata, X], axis=1).reset_index(drop=True)
        spark.createDataFrame(output).write.mode("overwrite").parquet(
            output_directory + name + ".parquet")
        print(name, "rows:", len(rows), "default rate:", round(rows.label.mean(), 3))
    print("OOT cutoff:", str(pd.Timestamp(oot_cutoff).date()))


if __name__ == "__main__":
    spark = pyspark.sql.SparkSession.builder \
        .appName("model_train") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_model_splits("datamart/gold/feature_store/", "datamart/gold/label_store/",
                         "datamart/model_data/", "model_bank/", spark)

    spark.stop()
