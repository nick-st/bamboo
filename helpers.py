import os
from copy import deepcopy
inlib = os.getcwd() + '/indata'
eda = os.getcwd() + '/EDA'
saved_models = os.getcwd() + '/saved_models'

def default_flag(x):
    """
    default_flg _summary_

    _extended_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    if x in ['Charged Off','Default']:
        return 1
    else:
        return 0
    
# variable transformations
def transform_to_categorical(df
, variable_list
, datatype = 'category'):
    
    """
    Transforms the variables in the variable_list to categorical variables. 

    _extended_summary_

    Returns:
        dataframe: The transformed dataframe.
    """
    
    df_out = deepcopy(df)
    
    for col in variable_list:
        df_out[col] = df_out[col].astype(datatype)
        
    print(df_out.dtypes)
        
    return df_out

def find_columns_with_one_value(df):
    """
    find_columns_with_one_value _summary_

    _extended_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    columns_with_one_value = []
    for column in df.columns:
        if df[column].nunique() == 1:
            columns_with_one_value.append(column)
    return columns_with_one_value


columns_to_drop_ids = [
  'id' # not used in the model, later might be used for tracking
, 'policy_code' # policy_code is always 1
, 'grade' # grade is included in sub_grade
, 'member_id' # not used in the model, later might be used for tracking
, 'issue_d' # TODO:  date might be used for additional features
, 'next_pymnt_d' # TODO: date might be used for additional features
, 'zip_code' # Zip code is also dropped, to avoid any bias
, 'url' # not used in the model, later might be used for tracking
, 'desc' # TODO: date might be used for additional features 
, 'sec_app_earliest_cr_line' # TODO:  date might be used for additional features
, 'earliest_cr_line' # TODO:  date might be used for additional features
, 'last_pymnt_d' # TODO:  date might be used for additional features
, 'last_credit_pull_d' # TODO:  date might be used for additional features
, 'settlement_date' # TODO:  date might be used for additional features
, 'debt_settlement_flag_date' # TODO:  date might be used for additional features
, 'hardship_start_date' # TODO:  date might be used for additional features
, 'hardship_end_date' # TODO:  date might be used for additional features
, 'payment_plan_start_date' # TODO:  date might be used for additional features
]

extra_columns_to_drop = [
  'recoveries'
, 'loan_amnt'
# , 'funded_amnt'
, 'funded_amnt_inv'
, 'collection_recovery_fee'
, 'out_prncp'
, 'out_prncp_inv'
, 'total_pymnt'
, 'total_pymnt_inv'
, 'total_rec_prncp'
, 'total_rec_int'
, 'total_rec_late_fee'
, 'last_pymnt_amnt'
, 'settlement_amount'
, 'settlement_percentage'
, 'settlement_term'
, 'hardship_amount'
, 'hardship_length' # Current derogatory item - not useful for modelling
, 'hardship_dpd'
, 'hardship_payoff_balance_amount'
, 'hardship_last_payment_amount'
, 'hardship_type' # Current derogatory item - not useful for modelling
, 'hardship_reason'
, 'hardship_status'
, 'deferral_term' # Current derogatory item - not useful for modelling
, 'hardship_loan_status'
, 'hardship_flag'
, 'debt_settlement_flag'
, 'debt_settlement_flag_date'
, 'settlement_status'
, 'settlement_date'
, 'settlement_amount'
, 'settlement_percentage'
, 'settlement_term'
]
