#!/usr/bin/env python
# coding: utf-8

# In[1]:


def cash_transform(cash,inplace=False):
    
    cash['pos_cash_paid_late'] = (cash['SK_DPD'] > 0).astype(int)
    cash['pos_cash_paid_late_with_tolerance'] = (cash['SK_DPD_DEF'] > 0).astype(int)
    
    def fix_skew_months(X):
        mask = X > 0
        X[mask] = np.NaN
        X = np.log(1+np.max(X)-X)
        return -X

    cash['MONTHS_BALANCE'] = fix_skew_months(cash['MONTHS_BALANCE'])
    cash['CNT_INSTALMENT'] = np.log1p(cash['CNT_INSTALMENT'])
    cash['CNT_INSTALMENT_FUTURE'] = np.log1p(cash['CNT_INSTALMENT_FUTURE'])
    
    return cash


# In[2]:


def cashAppsFeaturesAggregater(df, inplace=False):
    # pure state-less transformations 
    if inplace:
        df_new = df
    else:
        df_new = df.copy()
        
    aggr_df = pd.DataFrame({'SK_ID_CURR': df_new['SK_ID_CURR'].unique()}) 
    
    agg_dict = {
        'MONTHS_BALANCE': ["min", "max", "mean", "sum", "var"],
        'CNT_INSTALMENT': ["min", "max", "mean", "sum", "var"],
        'CNT_INSTALMENT_FUTURE': ["min", "max", "mean", "sum", "var"],
        'SK_DPD':["min", "max", "mean", "sum", "var"],
        'SK_DPD_DEF':["min", "max", "mean", "sum", "var"],
        'pos_cash_paid_late':["mean"],
        'pos_cash_paid_late_with_tolerance':["mean"]
        }
    
    X = df_new.groupby(["SK_ID_CURR"], as_index=False).agg(agg_dict)
    X.columns = X.columns.map(lambda col: '_'.join([x for x in col if x != '']))
    aggr_df = aggr_df.merge(X, how='left', on='SK_ID_CURR')
    
    return aggr_df


# In[3]:


def install_transform(install,inplace=False):
    
    install['installment_payment_diff'] = install['AMT_INSTALMENT'] - install['AMT_PAYMENT']
    install['installment_paid_in_full'] = np.where(install['installment_payment_diff']<= 0, 1, 
                                                   np.where(install['installment_payment_diff']>100.00,0,1))
    
    install['installment_days_diff'] = install['DAYS_INSTALMENT'] - install['DAYS_ENTRY_PAYMENT']
    install['installment_paid_in_time'] = np.where(install['installment_days_diff']>= 0, 1, 0)
    
    install['install_version'] = (install['NUM_INSTALMENT_VERSION'] > 0).astype(int)
    
    def left_skew_days(X):
        mask = X > 0
        X[mask] = np.NaN
        X = np.log(1+np.max(X)-X)
        return -X

    left_skewed = ['DAYS_INSTALMENT','DAYS_ENTRY_PAYMENT']
    install[left_skewed] = left_skew_days(install[left_skewed])
    install['NUM_INSTALMENT_NUMBER'] = np.log1p(install['NUM_INSTALMENT_NUMBER'])
    
    return install


# In[ ]:


def instlmntAppsFeaturesAggregater(df, inplace=False):
    # pure state-less transformations 
    if inplace:
        df_new = df
    else:
        df_new = df.copy()
        
    aggr_df = pd.DataFrame({'SK_ID_CURR': df_new['SK_ID_CURR'].unique()})
    
    # Compute min, max, min values
    agg_dict = {
        'NUM_INSTALMENT_VERSION':["min", "max", "mean", "sum", "var"],
       'NUM_INSTALMENT_NUMBER':["min", "max", "mean", "sum", "var"],
        'DAYS_INSTALMENT':["min", "max", "mean", "sum", "var"], 
        'DAYS_ENTRY_PAYMENT':["min", "max", "mean", "sum", "var"],
       'AMT_INSTALMENT':["min", "max", "mean", "sum", "var"], 
        'AMT_PAYMENT':["min", "max", "mean", "sum", "var"],
        'installment_payment_diff':["min", "max", "mean", "sum", "var"],
       'installment_paid_in_full':["mean"], 
        'installment_days_diff':["min", "max", "mean", "sum", "var"],
       'installment_paid_in_time':["mean"],
        'install_version':["mean"]
            } 
    X = df_new.groupby(["SK_ID_CURR"], as_index=False).agg(agg_dict)
    X.columns = X.columns.map(lambda col: '_'.join([x for x in col if x != '']))
    aggr_df = aggr_df.merge(X, how='left', on='SK_ID_CURR')
    
    return aggr_df


# In[4]:


def credit_transform(credit,inplace=False):
    
       # Amount used from limit
    credit['limit_use'] = credit['AMT_BALANCE'] / credit['AMT_CREDIT_LIMIT_ACTUAL']
    # Current payment / Min payment
    credit['payment_div_min'] = credit['AMT_PAYMENT_CURRENT'] / credit['AMT_INST_MIN_REGULARITY']
    # Late payment <-- 'CARD_IS_DPD'
    credit['late_payment'] = credit['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    # How much drawing of limit
    credit['drawing_limit_ratio'] = credit['AMT_DRAWINGS_ATM_CURRENT'] / credit['AMT_CREDIT_LIMIT_ACTUAL']
    
    def right_skew(X): return np.log1p(X)
    
    right_skewed = ['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL','AMT_RECEIVABLE_PRINCIPAL','AMT_RECIVABLE',
                   'AMT_TOTAL_RECEIVABLE','CNT_INSTALMENT_MATURE_CUM']
    credit[right_skewed] = right_skew(credit[ right_skewed])
    
    return credit


# In[ ]:


def creditAppsFeaturesAggregater(df, inplace=False):
    # pure state-less transformations 
    if inplace:
        df_new = df
    else:
        df_new = df.copy()
        
    aggr_df = pd.DataFrame({'SK_ID_CURR': df_new['SK_ID_CURR'].unique()})
    
    # Compute min, max, min values
    agg_dict = {
       'AMT_BALANCE':["min", "max", "mean", "sum", "var"],
       'AMT_CREDIT_LIMIT_ACTUAL':["min", "max", "mean", "sum", "var"], 
        'AMT_DRAWINGS_ATM_CURRENT':["min", "max", "mean", "sum", "var"],
       'AMT_DRAWINGS_CURRENT':["min", "max", "mean", "sum", "var"], 
        'AMT_DRAWINGS_OTHER_CURRENT':["min", "max", "mean", "sum", "var"],
       'AMT_DRAWINGS_POS_CURRENT':["min", "max", "mean", "sum", "var"], 
        'AMT_INST_MIN_REGULARITY':["min", "max", "mean", "sum", "var"],
       'AMT_PAYMENT_CURRENT':["min", "max", "mean", "sum", "var"], 
        'AMT_PAYMENT_TOTAL_CURRENT':["min", "max", "mean", "sum", "var"],
       'AMT_RECEIVABLE_PRINCIPAL':["min", "max", "mean", "sum", "var"], 
        'AMT_RECIVABLE':["min", "max", "mean", "sum", "var"], 
        'AMT_TOTAL_RECEIVABLE':["min", "max", "mean", "sum", "var"],
       'CNT_DRAWINGS_ATM_CURRENT':["min", "max", "mean", "sum", "var"],
        'CNT_DRAWINGS_CURRENT':["min", "max", "mean", "sum", "var"],
       'CNT_DRAWINGS_OTHER_CURRENT':["min", "max", "mean", "sum", "var"],
        'CNT_DRAWINGS_POS_CURRENT':["min", "max", "mean", "sum", "var"],
       'CNT_INSTALMENT_MATURE_CUM':["min", "max", "mean", "sum", "var"],
        'SK_DPD':["min", "max", "mean", "sum", "var"],
       'SK_DPD_DEF':["min", "max", "mean", "sum", "var"],
        'limit_use':["min", "max", "mean", "sum", "var"],
        'payment_div_min':["min", "max", "mean", "sum", "var"], 
        'late_payment':["mean"],
       'drawing_limit_ratio':["min", "max", "mean", "sum", "var"]
            } 
    X = df_new.groupby(["SK_ID_CURR"], as_index=False).agg(agg_dict)
    X.columns = X.columns.map(lambda col: '_'.join([x for x in col if x != '']))
    aggr_df = aggr_df.merge(X, how='left', on='SK_ID_CURR')
    
    return aggr_df

