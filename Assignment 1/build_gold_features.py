"""Build gold feature partitions from existing silver data (without rerunning bronze)."""
from pathlib import Path
from pyspark.sql import SparkSession
from utils import data_processing_gold_table as gold


def main():
    spark = SparkSession.builder.master("local[2]").appName("gold-features").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.shuffle.partitions", "4")
    try:
        for table in ["fe_attr", "fe_fin", "fe_click"]:
            source = Path("datamart/silver") / table
            target = Path("datamart/gold") / table
            target.mkdir(parents=True, exist_ok=True)
            process = getattr(gold, f"process_{table}_gold_table")
            for partition in sorted(source.glob(f"silver_{table}_*.parquet")):
                date = partition.stem.removeprefix(f"silver_{table}_").replace("_", "-")
                process(date, str(source) + "/", str(target) + "/", spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
