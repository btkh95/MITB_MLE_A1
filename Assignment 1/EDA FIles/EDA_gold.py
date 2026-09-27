# %% [markdown]
# # Gold table exploration
# Run all cells in the project Jupyter container. Reads existing gold Parquet outputs only.
# Counts describe customer snapshots, not independent people. Labels describe loan outcomes.
# Change the dates below to explore a smaller period. Ratios are diagnostics, not proof of errors.

# %%
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd() if Path('datamart').exists() else Path.cwd() / 'Assignment 1'
START_DATE = None  # Example: '2023-01-01'
END_DATE = None    # Example: '2024-12-01'
RATIO_LOW, RATIO_HIGH = 0.5, 1.5  # Exploration thresholds; not cleaning rules
con = duckdb.connect()
tables = {}
for name, folder in [('labels', 'label_store'), ('attributes', 'fe_attr'),
                     ('financials', 'fe_fin'), ('clickstream', 'fe_click')]:
    files = sorted(str(p) for p in (ROOT / 'datamart/gold' / folder).glob('*/part-*.parquet'))
    if not files:
        raise FileNotFoundError(f'No gold data for {folder}. Run build_gold_features.py first; labels require main.py.')
    frame = con.read_parquet(files, union_by_name=True).df()
    frame['snapshot_date'] = pd.to_datetime(frame['snapshot_date'])
    if START_DATE:
        frame = frame[frame.snapshot_date >= pd.Timestamp(START_DATE)]
    if END_DATE:
        frame = frame[frame.snapshot_date <= pd.Timestamp(END_DATE)]
    tables[name] = frame
labels, attr, fin, click = [tables[name] for name in ['labels', 'attributes', 'financials', 'clickstream']]
plt.rcParams.update({'figure.figsize': (10, 4), 'axes.grid': True, 'grid.alpha': 0.2})

# %% [markdown]
# ## Table coverage and missing values
# Duplicate keys are diagnostics: investigate before joining tables. Empty early clickstream
# partitions have no rows because only activity strictly before the snapshot is included.

# %%
overview = []
for name, df in tables.items():
    keys = ['loan_id', 'snapshot_date'] if name == 'labels' else ['Customer_ID', 'snapshot_date']
    overview.append({'table': name, 'rows': len(df), 'customers': df.Customer_ID.nunique(),
                     'columns': len(df.columns), 'first_date': df.snapshot_date.min(),
                     'last_date': df.snapshot_date.max(), 'duplicate_key_rows': int(df.duplicated(keys).sum())})
display(pd.DataFrame(overview))
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, (name, df) in zip(axes.flat, tables.items()):
    missing = df.isna().mean().mul(100).sort_values(ascending=False).head(10)
    missing.sort_values().plot.barh(ax=ax, title=f'{name}: top missing fields')
    ax.set_xlabel('Missing rows (%)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Monthly in-hand salary versus annual income
# Ratio = 12 × Monthly_Inhand_Salary / Annual_Income. A ratio of 1 means exact agreement.
# Gross/net definitions, deductions or additional income can explain differences; do not
# automatically replace either field. Only finite, positive pairs are compared here.
# The low/high thresholds are adjustable investigation flags, not validated business rules.

# %%
income = fin[['Customer_ID', 'snapshot_date', 'Annual_Income', 'Monthly_Inhand_Salary']].copy()
valid = (np.isfinite(income.Annual_Income) & np.isfinite(income.Monthly_Inhand_Salary)
         & income.Annual_Income.gt(0) & income.Monthly_Inhand_Salary.gt(0))
income = income.loc[valid].copy()
income['annualized_salary'] = income.Monthly_Inhand_Salary * 12
income['ratio'] = income.annualized_salary / income.Annual_Income
income['relative_gap'] = income.ratio - 1
income['extreme_ratio'] = (income.ratio < RATIO_LOW) | (income.ratio > RATIO_HIGH)
income_summary = pd.Series({
    'financial_rows': len(fin), 'valid_positive_pairs': len(income),
    'excluded_missing_nonpositive_or_nonfinite': len(fin) - len(income),
    'median_ratio': income.ratio.median(),
    'within_10_percent_count': int(income.ratio.between(0.9, 1.1).sum()),
    'within_10_percent_pct': income.ratio.between(0.9, 1.1).mean() * 100,
    'low_ratio_count': int((income.ratio < RATIO_LOW).sum()),
    'high_ratio_count': int((income.ratio > RATIO_HIGH).sum()),
    'extreme_ratio_pct': income.extreme_ratio.mean() * 100,
    'customers_with_extreme_ratio': income.loc[income.extreme_ratio, 'Customer_ID'].nunique(),
})
display(income_summary.to_frame('value'))
display(income[['Annual_Income', 'Monthly_Inhand_Salary', 'ratio']].describe(percentiles=[.01, .05, .5, .95, .99]))
display(income.sort_values('ratio').head(10))
display(income.sort_values('ratio', ascending=False).head(10))

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sample = income.sample(min(10000, len(income)), random_state=42)
axes[0].scatter(sample.Annual_Income, sample.annualized_salary, s=7, alpha=.2)
if len(income):
    lower = min(income.Annual_Income.min(), income.annualized_salary.min())
    upper = max(income.Annual_Income.max(), income.annualized_salary.max())
    axes[0].plot([lower, upper], [lower, upper], '--', color='black', label='Equal amounts')
axes[0].set(xscale='log', yscale='log', xlabel='Reported annual income',
            ylabel='Monthly in-hand salary × 12', title='Income comparison (log axes; up to 10,000 rows)')
axes[1].hist(np.log10(income.ratio), bins=60)
axes[1].axvline(0, color='black', linestyle='--')
axes[1].set(xlabel='log10(ratio): −1 = 0.1×, 0 = 1×, 1 = 10×', ylabel='Rows', title='Full ratio distribution')
plt.tight_layout()
plt.show()
monthly_income = income.groupby('snapshot_date').agg(rows=('ratio', 'size'),
    median_ratio=('ratio', 'median'), extreme_pct=('extreme_ratio', lambda s: s.mean() * 100))
display(monthly_income)

# %% [markdown]
# ## Investigate unusual income within a customer
# These cross-date medians are EDA diagnostics only. They use the selected date range,
# so must not be copied into model features or used to repair past records with future data.

# %%
observations = income.groupby('Customer_ID').size()
print('Customers with multiple financial snapshots:', int((observations > 1).sum()))
if not (observations > 1).any():
    print('Each customer has only one financial snapshot; within-customer history cannot validate these discrepancies.')
income['customer_median_income'] = income.groupby('Customer_ID').Annual_Income.transform('median')
income['income_vs_customer_median'] = income.Annual_Income / income.customer_median_income
display(income.loc[income.extreme_ratio].sort_values('income_vs_customer_median', ascending=False).head(20))
CUSTOMER_ID = income.sort_values('ratio').Customer_ID.iloc[0] if len(income) else None
# Replace CUSTOMER_ID with a customer you want to inspect.
display(income.loc[income.Customer_ID == CUSTOMER_ID].sort_values('snapshot_date'))

# %% [markdown]
# ## Financial features and quality flags
# Histograms show the middle 98% of finite values for readability; the summary retains all values.
# Missing ratios may reflect missing inputs or zero denominators. Negative disposable income is retained.

# %%
features = ['EMI_to_Salary_Ratio', 'Debt_to_Income', 'Disposable_Income', 'Credit_History_Months']
display(fin[features].describe(percentiles=[.01, .5, .99]))
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, column in zip(axes.flat, features):
    values = fin[column].replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = values.quantile([.01, .99]) if len(values) else (0, 0)
    ax.hist(values[values.between(lo, hi)], bins=40)
    ax.set(title=column + ' (1st–99th percentiles)', ylabel='Rows')
plt.tight_layout()
plt.show()
flags = [c for c in ['loan_count_suspicious', 'loan_types_missing', 'interest_rate_invalid'] if c in fin]
display(fin[flags].mean().mul(100).rename('flagged_pct').to_frame())
display(fin.loan_count_status.value_counts(dropna=False).rename('rows').to_frame())

# %% [markdown]
# ## Customer attributes
# These distributions count snapshots: a customer appearing in several months contributes several rows.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(attr.Age.dropna(), bins=30)
axes[0].set(title='Age', xlabel='Years', ylabel='Rows')
attr.Occupation.value_counts().head(15).sort_values().plot.barh(ax=axes[1], title='Top occupations')
plt.tight_layout()
plt.show()
display(attr[['age_invalid', 'occupation_missing']].mean().mul(100).rename('flagged_pct').to_frame())

# %% [markdown]
# ## Clickstream coverage and feature distributions
# has_clickstream is always 1 inside the click table; absent customers have no row there.
# Coverage below uses attribute customer/date keys as the population and includes absent customers.
# clickstream_days counts distinct observed dates; if source snapshots are monthly, it counts months observed.

# %%
population = attr[['Customer_ID', 'snapshot_date']].drop_duplicates()
activity = click[['Customer_ID', 'snapshot_date']].drop_duplicates().assign(has_history=1)
coverage = population.merge(activity, on=['Customer_ID', 'snapshot_date'], how='left', validate='one_to_one')
coverage['has_history'] = coverage.has_history.fillna(0)
monthly_coverage = coverage.groupby('snapshot_date').agg(customers=('Customer_ID', 'size'),
    coverage_pct=('has_history', lambda s: s.mean() * 100))
display(monthly_coverage)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
monthly_coverage.coverage_pct.plot(ax=axes[0], marker='o', title='Customers with prior clickstream')
axes[0].set_ylabel('Coverage (%)')
axes[0].set_ylim(0, 105)
axes[1].hist(click.clickstream_days.dropna(), bins=30)
axes[1].set(title='Historical observed-date counts', xlabel='Distinct dates', ylabel='Customer snapshots')
plt.tight_layout()
plt.show()
display(click[[c for c in click if c.startswith('fe_')]].describe())

# %% [markdown]
# ## Labels over time
# Label date is the outcome observation date at month-on-book 6, not the feature date.
# Do not join labels to features on this date directly. Align features to the prediction/loan-start
# date using loan_id and loan_start_date from silver when building a training dataset.

# %%
label_summary = labels.groupby('snapshot_date').agg(loans=('label', 'size'), default_rate=('label', 'mean'))
display(label_summary)
display(labels.label.value_counts(dropna=False).rename('rows').to_frame())
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
label_summary.loans.plot(ax=axes[0], marker='o', title='Labeled loans')
label_summary.default_rate.mul(100).plot(ax=axes[1], marker='o', title='Default rate at MOB 6')
axes[1].set_ylabel('Default rate (%)')
plt.tight_layout()
plt.show()
