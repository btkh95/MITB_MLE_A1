"""Small synthetic integration check; run in the project's Spark container."""
import tempfile
from pathlib import Path

from pyspark.sql import SparkSession
from utils.data_processing_gold_table import (
    process_lms_gold_table, process_fe_attr_gold_table,
    process_fe_fin_gold_table, process_fe_click_gold_table,
)


def main():
    spark = SparkSession.builder.master("local[1]").appName("gold-check").getOrCreate()
    spark.conf.set("spark.sql.ansi.enabled", "true")
    spark.conf.set("spark.sql.shuffle.partitions", "1")
    spark.sparkContext.setLogLevel("ERROR")
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ["attr", "fin", "click", "lms", "gold"]:
                (root / name).mkdir()
            directory = lambda name: str(root / name) + "/"
            date = "2024-03-01"
            attr = spark.sql("""SELECT 'C1' Customer_ID, 25 Age, '25_' Age_raw,
                false age_invalid, cast(null as string) Occupation,
                date '2024-03-01' snapshot_date, '2024-03-01' snapshot_date_raw""")
            attr.write.parquet(directory("attr") + "silver_fe_attr_2024_03_01.parquet")
            result = process_fe_attr_gold_table(date, directory("attr"), directory("gold"), spark)
            row = result.first()
            assert row.Occupation == "Unknown" and row.occupation_missing
            assert "Age_raw" not in result.columns

            fin = spark.sql("""SELECT 'C1' Customer_ID, 12000.0 Annual_Income,
                1000.0 Monthly_Inhand_Salary, 200.0 Total_EMI_per_month,
                100.0 Amount_invested_monthly, 6000.0 Outstanding_Debt,
                -10.0 Monthly_Balance, 'Auto Loan, Auto Loan, Personal Loan' Type_of_Loan,
                false loan_types_missing, 'High_spent_Small_value_payments' Payment_Behaviour,
                cast(null as string) Credit_Mix, cast(null as string) Payment_of_Min_Amount""")
            from pyspark.sql import functions as F
            missing = (fin.withColumn("Customer_ID", F.lit("C2"))
                .withColumn("Annual_Income", F.lit(0.0))
                .withColumn("Monthly_Inhand_Salary", F.lit(0.0))
                .withColumn("Type_of_Loan", F.lit(None).cast("string"))
                .withColumn("loan_types_missing", F.lit(True))
                .withColumn("Payment_Behaviour", F.lit(None).cast("string")))
            fin.unionByName(missing).write.parquet(directory("fin") + "silver_fe_fin_2024_03_01.parquet")
            result = process_fe_fin_gold_table(date, directory("fin"), directory("gold"), spark)
            rows = {row.Customer_ID: row for row in result.collect()}
            assert rows["C1"].EMI_to_Salary_Ratio == 0.2
            assert rows["C1"].Debt_to_Income == 0.5
            assert rows["C1"].Disposable_Income == 700
            assert rows["C1"].Loan_Auto_Loan == 2
            assert rows["C1"].Payments_Size == "Small"
            assert rows["C1"].Monthly_Balance_log1p is None
            assert rows["C2"].EMI_to_Salary_Ratio is None
            assert rows["C2"].Debt_to_Income is None
            assert rows["C2"].Loan_Auto_Loan is None
            assert rows["C2"].Payments_Size == "Unknown"

            click = spark.sql("""SELECT * FROM VALUES
                ('C1', date '2024-01-01', 2), ('C1', date '2024-02-01', 4),
                ('C1', date '2024-02-01', 6), ('C1', date '2024-03-01', 100),
                ('C1', date '2024-04-01', 200), ('C2', date '2024-03-01', 10)
                AS clicks(Customer_ID, snapshot_date, fe_1)""")
            click.write.parquet(directory("click") + "silver_fe_click_2024_01_01.parquet")
            result = process_fe_click_gold_table(date, directory("click"), directory("gold"), spark)
            rows = result.collect()
            assert len(rows) == 1 and rows[0].fe_1_mean == 4
            assert rows[0].clickstream_days == 2 and rows[0].has_clickstream == 1
            assert str(rows[0].snapshot_date) == date
            empty = process_fe_click_gold_table("2023-01-01", directory("click"), directory("gold"), spark)
            assert empty.count() == 0 and "fe_1_mean" in empty.columns

            lms = spark.sql("""SELECT * FROM VALUES
                ('L1', 'C1', 6, 30, date '2024-03-01'),
                ('L2', 'C2', 6, 29, date '2024-03-01'),
                ('L3', 'C3', 5, 60, date '2024-03-01')
                AS loans(loan_id, Customer_ID, mob, dpd, snapshot_date)""")
            lms.write.parquet(directory("lms") + "silver_loan_daily_2024_03_01.parquet")
            result = process_lms_gold_table(date, directory("lms"), directory("gold"), spark, 30, 6)
            assert {row.loan_id: row.label for row in result.collect()} == {"L1": 1, "L2": 0}
            print("PASS: gold transformations, invalid inputs, date cutoff, empty history and labels")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
