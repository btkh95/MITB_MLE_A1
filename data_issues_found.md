- about 66 customers have a Num_of_Loan that does not match the number of loans listed in Type_of_Loan
Decision: keep these customers, set Num_of_Loan to null and flag them with loan_count_suspicious (silver). Their other features are still usable, and the listed loan types are still counted in the gold Loan_* columns.

- Amount_invested_monthly uses the placeholder __10000__ (558 rows)
Decision: set 10000 to null in silver; it is a placeholder, not a real amount.

- Monthly_Balance has one placeholder value of about -3.3e26
Decision: set it to null in silver; it would otherwise distort scaling for the whole column.

- Interest_Rate, Num_Bank_Accounts, Num_Credit_Card, Num_Credit_Inquiries and Num_of_Delayed_Payment have extreme values far above the normal range
Decision: keep the valid ranges in static/valid_ranges.yaml; values outside them are set to null and flagged as <column>_invalid in silver.

- Total_EMI_per_month is above monthly salary for about 400 rows, and Annual_Income is above 30x monthly salary for about 110 rows
Decision: check these against Monthly_Inhand_Salary using the ratio ranges in static/valid_ranges.yaml; values outside them are set to null and flagged as <column>_invalid in silver.
