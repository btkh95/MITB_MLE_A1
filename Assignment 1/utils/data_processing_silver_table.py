import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_lms_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_fe_attr_silver_table(snapshot_date_str, bronze_fe_attr_directory, silver_fe_attr_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_fe_attr_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    # drop PII columns - will never be used in modeling, and should not be stored in silver table
    pii_columns = read_yaml_file("static/dir.yaml")["pii"]
    df = df.drop(*pii_columns)

    # print out the dropped PII columns and the row count after dropping
    print('dropped PII columns:', pii_columns, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
        "Age": IntegerType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # extract only numbers from age column and insert nulls for age, with conditional negative sign, filter out that that are not between 0 and 100, and convert to integer
    df = df.withColumn("Age", F.when(F.col("Age").rlike(r"^\d+$"), F.col("Age").cast(IntegerType())).otherwise(None)).filter(F.col("Age").between(0, 100))
    
    # Clean Occupation that are underlines only
    df = df.withColumn("Occupation", F.when(F.col("Occupation").rlike(r"^[a-zA-Z_]+$"), F.col("Occupation")).otherwise(None))

    # save silver table - IRL connect to database to write
    partition_name = "silver_fe_attr_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_fe_attr_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_fe_fin_silver_table(snapshot_date_str, bronze_fe_fin_directory, silver_fe_fin_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_fe_fin_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": IntegerType(),
        "Monthly_Inhand_Salary": IntegerType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": FloatType(),
        "Type_of_Loan": FloatType(),
        "Delay_from_due_date": FloatType(),
        "Num_of_Delayed_Payment": FloatType(),
        "Changed_Credit_Limit": DateType(),
        "Num_Credit_Inquiries": DateType(),
        "Credit_Mix": DateType(),
        "Outstanding_Debt": DateType(),
        "Credit_Utilization_Ratio": DateType(),
        "Credit_History_Age": DateType(),
        "Payment_of_Min_Amount": DateType(),
        "Total_EMI_per_month": DateType(),
        "Amount_invested_monthly": DateType(),
        "Payment_Behaviour": DateType(),
        "Monthly_Balance": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data - split Type_of_Loan delimiters ", " and ", and " -> Total, Debt, Personal, Revolving, Mortgage, Auto, Home_Equity, Other
    df = df.withColumn("Type_of_Loan", F.regexp_replace(col("Type_of_Loan"), ", and ", ", "))
    
    # remove "_" in num_of_loan
    df = df.withColumn("num_of_loan", F.regexp_replace(col("num_of_loan"), "_", " "))

    # filter out interest rate that are negative or greater than 34
    df = df.filter((col("Interest_Rate") >= 0) & (col("Interest_Rate") <= 34))

    # Credit Mix: replace those that do not have the valid values with nulls
    valid_credit_mix_values = ["Standard", "Good", "Bad"]
    df = df.withColumn("Credit_Mix", F.when(col("Credit_Mix").isin(valid_credit_mix_values), col("Credit_Mix")).otherwise(None))

    # save silver table - IRL connect to database to write
    partition_name = "silver_fe_fin_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_fe_fin_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_lms_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df
