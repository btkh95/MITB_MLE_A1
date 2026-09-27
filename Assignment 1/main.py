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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table
import utils.data_processing_model_data
from utils.check_data_files import check_data_files

import yaml

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# read in config file for directories and filenames
with open("static/dir.yaml") as file:
    config = yaml.safe_load(file)

check_data_files(config)

##########
# Bronze #
##########

# create bronze datalake - loop through all bronze datasets in config file
for bronze_subdir in config["bronze"]["datasets"]:
    bronze_directory = "datamart/bronze/" + bronze_subdir + "/"

    if not os.path.exists(bronze_directory):
        os.makedirs(bronze_directory)

bronze_lms_directory = "datamart/bronze/lms/"
bronze_fe_attr_directory = "datamart/bronze/fe_attr/"
bronze_fe_fin_directory = "datamart/bronze/fe_fin/"
bronze_fe_click_directory = "datamart/bronze/fe_click/"

# run bronze backfill
for bronze_subdir in config["bronze"]["datasets"]:
    csv_file_path = "data/" + config["bronze"]["datasets"][bronze_subdir]["filename"]
    bronze_directory = "datamart/bronze/" + bronze_subdir + "/"

    for date_str in dates_str_lst:
        utils.data_processing_bronze_table.process_bronze_table(
            date_str, csv_file_path, bronze_directory, spark
        )

##########
# Silver #
##########

# create silver datalake
silver_loan_daily_directory = "datamart/silver/loan_daily/"
silver_fe_attr_directory = "datamart/silver/fe_attr/"
silver_fe_fin_directory = "datamart/silver/fe_fin/"
silver_fe_click_directory = "datamart/silver/fe_click/"

for silver_directory in [silver_loan_daily_directory, silver_fe_attr_directory,
                         silver_fe_fin_directory, silver_fe_click_directory]:
    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_lms_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
    utils.data_processing_silver_table.process_fe_attr_silver_table(date_str, bronze_fe_attr_directory, silver_fe_attr_directory, spark)
    utils.data_processing_silver_table.process_fe_fin_silver_table(date_str, bronze_fe_fin_directory, silver_fe_fin_directory, spark)
    utils.data_processing_silver_table.process_fe_click_silver_table(date_str, bronze_fe_click_directory, silver_fe_click_directory, spark)

########
# Gold #
########

# create gold datalake
gold_label_store_directory = "datamart/gold/label_store/"
gold_feature_store_directory = "datamart/gold/feature_store/"
gold_fe_attr_directory = gold_feature_store_directory + "fe_attr/"
gold_fe_fin_directory = gold_feature_store_directory + "fe_fin/"
gold_fe_click_directory = gold_feature_store_directory + "fe_click/"

for gold_directory in [gold_label_store_directory, gold_feature_store_directory,
                       gold_fe_attr_directory, gold_fe_fin_directory, gold_fe_click_directory]:
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_lms_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)
    utils.data_processing_gold_table.process_fe_attr_gold_table(date_str, silver_fe_attr_directory, gold_fe_attr_directory, spark)
    utils.data_processing_gold_table.process_fe_fin_gold_table(date_str, silver_fe_fin_directory, gold_fe_fin_directory, spark)
    utils.data_processing_gold_table.process_fe_click_gold_table(date_str, silver_fe_click_directory, gold_fe_click_directory, spark)

# assemble the feature store after every monthly feature partition is available
utils.data_processing_gold_table.process_features_gold_table(gold_feature_store_directory, spark)

# Decision: keep main.py as the pipeline orchestrator. Use the latest 20% of
# application months for OOT; stratify an 80/20 train/test split on older loans.
# The helper uses training means (including clickstream means from clickers),
# training medians, one-hot encoding and StandardScaler() on train only.
utils.data_processing_model_data.process_model_splits(
    gold_feature_store_directory, gold_label_store_directory,
    "datamart/gold/post_split/", spark, oot_fraction=0.2, test_size=0.2, random_state=55)

folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, 'gold_label_store_*.parquet'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()

# end spark session
spark.stop()



    
