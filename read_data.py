import pandas as pd
import numpy as np
from helpers import inlib, default_flag, eda
from sklearn.model_selection import train_test_split
from helpers import transform_to_categorical, find_columns_with_one_value, columns_to_drop_ids, extra_columns_to_drop
df = pd.read_csv(f'{inlib}/accepted_2007_to_2018Q4.csv', low_memory=True)
dd = pd.read_csv(f'{inlib}/LCDataDictionary.csv')
df['loan_status'].value_counts(dropna=False)

# loan_status
# Fully Paid                                             1076751
# Current                                                 878317
# Charged Off                                             268559
# Late (31-120 days)                                       21467
# In Grace Period                                           8436
# Late (16-30 days)                                         4349
# Does not meet the credit policy. Status:Fully Paid        1988
# Does not meet the credit policy. Status:Charged Off        761
# Default                                                     40
# NaN                                                         33
# https://www.lendingclub.com/help/investing-faq/what-do-the-different-note-statuses-mean

exclusions = ['Does not meet the credit policy. Status:Fully Paid','Does not meet the credit policy. Status:Charged Off',np.NaN]
df = df[~df["loan_status"].isin(exclusions)]
df['loan_status'].value_counts(dropna=False)
df['default_flag'] = df['loan_status'].apply(default_flag)

print(pd.crosstab(df['loan_status'], df['default_flag'], margins=True, dropna=False))
print(pd.crosstab(df['title'], df['purpose'], dropna=False))
print(pd.crosstab(df['title'], df['purpose'], dropna=False))

# NOTE: drop columns related to recoveries, since they are not available at the time of the loan
# NOTE: Drop columns like dates, which are not useful for the model (they will be useful for extra derived features)
# NOTE: Zip code is also dropped, to avoid any bias
# TODO: Hardhip period derive
one_value = find_columns_with_one_value(df)
[print(df[i].value_counts(dropna = False)) for i in one_value]
pd.crosstab(df['sub_grade'], df['grade'], margins=True)
columns_to_drop = columns_to_drop_ids + extra_columns_to_drop + one_value
df.drop(columns_to_drop, axis=1, inplace=True)

list(df.columns)
# NOTE: Extract the rest of the data from loan_status column and drop it
df['late_31_120d'] = np.where(df['loan_status'] == 'Late (31-120 days)', 1, 0)
df['late_16_30d'] = np.where(df['loan_status'] == 'Late (16-30 days)', 1, 0)
df['grace_period'] = np.where(df['loan_status'] == 'In Grace Period', 1, 0)  
df.drop('loan_status', axis=1, inplace=True)

# TODO: More complex split is needed, so that we make sure to keep the full history of each account/id
df, _ = train_test_split(df, test_size=0.95, stratify=df['default_flag'])

# NOTE: Transform to categorical
object_fields = list(df.select_dtypes(include='object').columns)
df = transform_to_categorical(df, object_fields)
# Export to parquet
df.to_parquet(f'{inlib}/df.parquet', index=False)

# A few manual checks below
set(pd.read_parquet(f'{inlib}/df2.parquet').columns).intersection(set(dd['LoanStatNew']))