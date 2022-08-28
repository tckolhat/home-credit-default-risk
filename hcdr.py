from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, f1_score
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import warnings
import pprint
# warnings.filterwarnings('ignore')


def load_data(in_path, name):
    df = pd.read_csv(in_path)
    return df


def load_datasets(DATA_DIR, ds_names):
    datasets = {}
    for ds_name in ds_names:
        datasets[ds_name] = load_data(os.path.join(
            DATA_DIR, f'{ds_name}.csv'), ds_name)
    return datasets


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


# Days Tranformation


def transform_days(X):
    mask = X > 0
    X[mask] = np.NaN
    # return np.log1p(-1*X)
    return -X


def preprocessing_transformations(df, inplace=False, impute_zero=()):
    # pure state-less transformations
    if inplace:
        df_new = df
    else:
        df_new = df.copy()

    right_skewed = ['AMT_ANNUITY']
    left_skewed = []
    days = ['DAYS_EMPLOYED']

    def transform_left_skewed(X): return np.log(1+np.max(X)-X)

    df_new[right_skewed] = np.log1p(df[right_skewed])
    df_new[left_skewed] = transform_left_skewed(df[left_skewed])
    df_new[days] = transform_days(df[days])

    # others
    df_new[impute_zero] = SimpleImputer(strategy='constant', fill_value=0).fit_transform(df_new[impute_zero])
    df_new['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
    df_new['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)
    df_new['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
    return df_new


def add_new_features(df, inplace=False):
    if inplace:
        X = df
    else:
        X = df.copy()
    X['annuity_income_percentage'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
    X['car_to_birth_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_BIRTH']
    X['car_to_employ_ratio'] = X['OWN_CAR_AGE'] / (1+X['DAYS_EMPLOYED'])
    X['children_ratio'] = X['CNT_CHILDREN'] / X['CNT_FAM_MEMBERS']
    X['credit_to_annuity_ratio'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
    X['credit_to_goods_ratio'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
    X['credit_to_income_ratio'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
    X['days_employed_percentage'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
    X['income_credit_percentage'] = X['AMT_INCOME_TOTAL'] / X['AMT_CREDIT']
    X['income_per_child'] = X['AMT_INCOME_TOTAL'] / (1 + X['CNT_CHILDREN'])
    X['income_per_person'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS']
    X['payment_rate'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
    X['phone_to_birth_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_BIRTH']
    X['phone_to_employ_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / (1+X['DAYS_EMPLOYED'])
    X['external_source_mean'] = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    X['cnt_non_child'] = X['CNT_FAM_MEMBERS'] - X['CNT_CHILDREN']
    X['child_to_non_child_ratio'] = X['CNT_CHILDREN'] / X['cnt_non_child']
    X['income_per_non_child'] = X['AMT_INCOME_TOTAL'] / X['cnt_non_child']
    X['credit_per_person'] = X['AMT_CREDIT'] / X['CNT_FAM_MEMBERS']
    X['credit_per_child'] = X['AMT_CREDIT'] / (1 + X['CNT_CHILDREN'])
    X['credit_per_non_child'] = X['AMT_CREDIT'] / X['cnt_non_child']

    return X


def cash_transform(cash, inplace=False):

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
        'SK_DPD': ["min", "max", "mean", "sum", "var"],
        'SK_DPD_DEF': ["min", "max", "mean", "sum", "var"],
        'pos_cash_paid_late': ["mean"],
        'pos_cash_paid_late_with_tolerance': ["mean"]
    }

    X = df_new.groupby(["SK_ID_CURR"], as_index=False).agg(agg_dict)
    X.columns = X.columns.map(lambda col: '_'.join([x for x in col if x != '']))
    aggr_df = aggr_df.merge(X, how='left', on='SK_ID_CURR')

    return aggr_df


def install_transform(install, inplace=False):

    install['installment_payment_diff'] = install['AMT_INSTALMENT'] - install['AMT_PAYMENT']
    install['installment_paid_in_full'] = np.where(install['installment_payment_diff'] <= 0, 1,
                                                   np.where(install['installment_payment_diff'] > 100.00, 0, 1))

    install['installment_days_diff'] = install['DAYS_INSTALMENT'] - install['DAYS_ENTRY_PAYMENT']
    install['installment_paid_in_time'] = np.where(install['installment_days_diff'] >= 0, 1, 0)

    install['install_version'] = (install['NUM_INSTALMENT_VERSION'] > 0).astype(int)

    def left_skew_days(X):
        mask = X > 0
        X[mask] = np.NaN
        X = np.log(1+np.max(X)-X)
        return -X

    left_skewed = ['DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']
    install[left_skewed] = left_skew_days(install[left_skewed])
    install['NUM_INSTALMENT_NUMBER'] = np.log1p(install['NUM_INSTALMENT_NUMBER'])

    return install


def instlmntAppsFeaturesAggregater(df, inplace=False):
    # pure state-less transformations
    if inplace:
        df_new = df
    else:
        df_new = df.copy()

    aggr_df = pd.DataFrame({'SK_ID_CURR': df_new['SK_ID_CURR'].unique()})

    # Compute min, max, min values
    agg_dict = {
        'NUM_INSTALMENT_VERSION': ["min", "max", "mean", "sum", "var"],
        'NUM_INSTALMENT_NUMBER': ["min", "max", "mean", "sum", "var"],
        'DAYS_INSTALMENT': ["min", "max", "mean", "sum", "var"],
        'DAYS_ENTRY_PAYMENT': ["min", "max", "mean", "sum", "var"],
        'AMT_INSTALMENT': ["min", "max", "mean", "sum", "var"],
        'AMT_PAYMENT': ["min", "max", "mean", "sum", "var"],
        'installment_payment_diff': ["min", "max", "mean", "sum", "var"],
        'installment_paid_in_full': ["mean"],
        'installment_days_diff': ["min", "max", "mean", "sum", "var"],
        'installment_paid_in_time': ["mean"],
        'install_version': ["mean"]
    }
    X = df_new.groupby(["SK_ID_CURR"], as_index=False).agg(agg_dict)
    X.columns = X.columns.map(lambda col: '_'.join([x for x in col if x != '']))
    aggr_df = aggr_df.merge(X, how='left', on='SK_ID_CURR')

    return aggr_df


def credit_transform(credit, inplace=False):

    # # Amount used from limit
    # credit['limit_use'] = credit['AMT_BALANCE'] / (1+credit['AMT_CREDIT_LIMIT_ACTUAL'])
    # # Current payment / Min payment
    # credit['payment_div_min'] = credit['AMT_PAYMENT_CURRENT'] / (1+credit['AMT_INST_MIN_REGULARITY'])
    # # Late payment <-- 'CARD_IS_DPD'
    # credit['late_payment'] = credit['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    # # How much drawing of limit
    # credit['drawing_limit_ratio'] = credit['AMT_DRAWINGS_ATM_CURRENT'] / (1+credit['AMT_CREDIT_LIMIT_ACTUAL'])

    def right_skew(X): return np.log1p(X)

    right_skewed = ['AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE',
                    'AMT_TOTAL_RECEIVABLE', 'CNT_INSTALMENT_MATURE_CUM']
    credit[right_skewed] = right_skew(credit[right_skewed])

    return credit


def creditAppsFeaturesAggregater(df, inplace=False):
    # pure state-less transformations
    if inplace:
        df_new = df
    else:
        df_new = df.copy()

    aggr_df = pd.DataFrame({'SK_ID_CURR': df_new['SK_ID_CURR'].unique()})

    # Compute min, max, min values
    agg_dict = {
        'AMT_BALANCE': ["min", "max", "mean", "sum", "var"],
        'AMT_CREDIT_LIMIT_ACTUAL': ["min", "max", "mean", "sum", "var"],
        'AMT_DRAWINGS_ATM_CURRENT': ["min", "max", "mean", "sum", "var"],
        'AMT_DRAWINGS_CURRENT': ["min", "max", "mean", "sum", "var"],
        'AMT_DRAWINGS_OTHER_CURRENT': ["min", "max", "mean", "sum", "var"],
        'AMT_DRAWINGS_POS_CURRENT': ["min", "max", "mean", "sum", "var"],
        'AMT_INST_MIN_REGULARITY': ["min", "max", "mean", "sum", "var"],
        'AMT_PAYMENT_CURRENT': ["min", "max", "mean", "sum", "var"],
        'AMT_PAYMENT_TOTAL_CURRENT': ["min", "max", "mean", "sum", "var"],
        'AMT_RECEIVABLE_PRINCIPAL': ["min", "max", "mean", "sum", "var"],
        'AMT_RECIVABLE': ["min", "max", "mean", "sum", "var"],
        'AMT_TOTAL_RECEIVABLE': ["min", "max", "mean", "sum", "var"],
        'CNT_DRAWINGS_ATM_CURRENT': ["min", "max", "mean", "sum", "var"],
        'CNT_DRAWINGS_CURRENT': ["min", "max", "mean", "sum", "var"],
        'CNT_DRAWINGS_OTHER_CURRENT': ["min", "max", "mean", "sum", "var"],
        'CNT_DRAWINGS_POS_CURRENT': ["min", "max", "mean", "sum", "var"],
        'CNT_INSTALMENT_MATURE_CUM': ["min", "max", "mean", "sum", "var"],
        # 'limit_use': ["min", "max", "mean", "sum", "var"],
        # 'payment_div_min': ["min", "max", "mean", "sum", "var"],
        # 'late_payment': ["mean"],
        # 'drawing_limit_ratio': ["min", "max", "mean", "sum", "var"]
    }
    X = df_new.groupby(["SK_ID_CURR"], as_index=False).agg(agg_dict)
    X.columns = X.columns.map(lambda col: '_'.join([x for x in col if x != '']))
    aggr_df = aggr_df.merge(X, how='left', on='SK_ID_CURR')

    return aggr_df


def bureauAppsFeaturesAggregater(df, inplace=False):
    # pure state-less transformations
    if inplace:
        df_new = df
    else:
        df_new = df.copy()

    aggr_df = pd.DataFrame({'SK_ID_CURR': df_new['SK_ID_CURR'].unique()})

    # Compute min, max, min values
    agg_ops = agg_ops = ["min", "max", "mean", "sum"]
    features = ['AMT_CREDIT_SUM', 'DAYS_CREDIT', 'DAYS_CREDIT_UPDATE', 'DAYS_CREDIT_ENDDATE']
    X = df_new.groupby(["SK_ID_CURR"], as_index=False).agg({ft: agg_ops for ft in features})
    X.columns = X.columns.map(lambda col: '_'.join([x for x in col if x != '']))
    aggr_df = aggr_df.merge(X, how='left', on='SK_ID_CURR')

    return aggr_df


def prevAppsFeaturesAggregater(df, inplace=False):
    # pure state-less transformations
    if inplace:
        df_new = df
    else:
        df_new = df.copy()

    # Sorted df by decsion day
    prev_applications_sorted = df_new.sort_values(
        ['SK_ID_CURR', 'DAYS_DECISION'])

    # Tranform days
    days = ['DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
            'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    df_new[days] = transform_days(df[days])

    aggr_df = pd.DataFrame({'SK_ID_CURR': df_new['SK_ID_CURR'].unique()})

    # Compute min, max, min values
    agg_ops = agg_ops = ["min", "max", "mean", "sum"]
    features = [
        'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE', 'CNT_PAYMENT',
        'HOUR_APPR_PROCESS_START', 'RATE_DOWN_PAYMENT', 'DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
        'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    X = df_new.groupby(["SK_ID_CURR"], as_index=False).agg({ft: agg_ops for ft in features})
    X.columns = X.columns.map(lambda col: '_'.join([x for x in col if x != '']))
    aggr_df = aggr_df.merge(X, how='left', on='SK_ID_CURR')

    # Previous Application Count
    prev_appl_count = df_new.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index()
    prev_appl_count.rename(index=str, columns={'SK_ID_PREV': 'previous_applications_count'}, inplace=True)
    aggr_df = aggr_df.merge(prev_appl_count, how='left', on='SK_ID_CURR')

    # Previous applications approved count
    df_new['prev_applications_approved'] = (df_new['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')
    approved_count = df_new.groupby(by=['SK_ID_CURR'])['prev_applications_approved'].sum().reset_index()
    aggr_df = aggr_df.merge(approved_count, how='left', on='SK_ID_CURR')

    # Previous applications refused count
    df_new['prev_applications_refused'] = (df_new['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
    refused_count = df_new.groupby(by=['SK_ID_CURR'])['prev_applications_refused'].sum().reset_index()
    aggr_df = aggr_df.merge(refused_count, how='left', on='SK_ID_CURR')

    # previous application invalid
    df_new['prev_applications_invalid'] = (df_new['NAME_CONTRACT_STATUS'] == 'Canceled').astype(
        'int') + (df_new['NAME_CONTRACT_STATUS'] == 'Unused offer').astype('int')
    invalid_count = df_new.groupby(by=['SK_ID_CURR'])['prev_applications_invalid'].sum().reset_index()
    aggr_df = aggr_df.merge(invalid_count, how='left', on='SK_ID_CURR')

    # Last application status(approved or rejected?)
    prev_applications_sorted['prevAppl_last_approved'] = (
        prev_applications_sorted['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')
    last_approved = prev_applications_sorted.groupby(by=['SK_ID_CURR'])['prevAppl_last_approved'].last().reset_index()
    aggr_df = aggr_df.merge(last_approved, how='left', on=['SK_ID_CURR'])
    return aggr_df


def make_prep_pipeline(num_selected=None, cat_selected=None):
    num_pipeline = Pipeline([
        ('new_features', FunctionTransformer(add_new_features)),
        ('selector', DataFrameSelector(num_selected)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('std_scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_selected)),
        #('imputer', SimpleImputer(strategy='most_frequent')),
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(sparse=False, handle_unknown="ignore"))
    ])

    data_prep_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
    ])
    return data_prep_pipeline


def load_process_data():
    # load data
    DATA_DIR = "../data"
    # ds_names = ("application_train", "application_test", "bureau","bureau_balance","credit_card_balance","installments_payments",
    #             "previous_application","POS_CASH_balance")
    ds_names = ("application_train", "bureau", "credit_card_balance", "installments_payments",
                "previous_application", "POS_CASH_balance")
    datasets = load_datasets(DATA_DIR, ds_names)
    print('loaded data')

    # Preparing data
    appl_train = datasets['application_train']
    prevData_aggr = prevAppsFeaturesAggregater(datasets['previous_application'])

    # bureau
    bureauData_aggr = bureauAppsFeaturesAggregater(datasets['bureau'])
    data_aggr = appl_train.merge(prevData_aggr, how='left', on=['SK_ID_CURR'])
    data_aggr = data_aggr.merge(bureauData_aggr, how='left', on=['SK_ID_CURR'])

    # cash
    cash = datasets['POS_CASH_balance']
    cashData_aggr = cashAppsFeaturesAggregater(cash_transform(cash))
    data_aggr = data_aggr.merge(cashData_aggr, how='left', on=['SK_ID_CURR'])
    install = datasets['installments_payments']
    instlmntData_aggr = instlmntAppsFeaturesAggregater(install_transform(install))
    data_aggr = data_aggr.merge(instlmntData_aggr, how='left', on=['SK_ID_CURR'])
    credit = datasets['credit_card_balance']
    creditData_aggr = creditAppsFeaturesAggregater(credit_transform(credit))

    data_aggr = data_aggr.merge(creditData_aggr, how='left', on=['SK_ID_CURR'])
    impute_zero = ['OWN_CAR_AGE', 'previous_applications_count', 'prev_applications_approved',
                   'prev_applications_refused', 'prev_applications_invalid', 'prevAppl_last_approved']
    processed_data = preprocessing_transformations(data_aggr, impute_zero=impute_zero)

    # training

    app_num_attribs = ['AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'AMT_CREDIT', 'REGION_POPULATION_RELATIVE',
                       'DAYS_EMPLOYED', 'DAYS_BIRTH', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_ID_PUBLISH',
                       'DAYS_REGISTRATION', 'DAYS_LAST_PHONE_CHANGE', 'OWN_CAR_AGE', 'OBS_30_CNT_SOCIAL_CIRCLE',
                       'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
                       'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',
                       'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
                       'HOUR_APPR_PROCESS_START']
    new_app_attribs = [
        'annuity_income_percentage', 'car_to_birth_ratio', 'car_to_employ_ratio', 'children_ratio',
        'credit_to_annuity_ratio', 'credit_to_goods_ratio', 'credit_to_income_ratio', 'days_employed_percentage',
        'income_credit_percentage', 'income_per_child', 'income_per_person', 'payment_rate', 'phone_to_birth_ratio',
        'phone_to_employ_ratio', 'external_source_mean', 'cnt_non_child', 'child_to_non_child_ratio',
        'income_per_non_child', 'credit_per_person', 'credit_per_child', 'credit_per_non_child']
    prev_aggr_attribs = prevData_aggr.columns.to_list()
    bureau_aggr_attribs = bureauData_aggr.columns.to_list()
    cash_columns = cashData_aggr.columns.to_list()
    install_columns = instlmntData_aggr.columns.to_list()
    credit_columns = creditData_aggr.columns.to_list()

    app_cat_attribs = [
        'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'HOUSETYPE_MODE',
        'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FLAG_MOBIL',
        'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'FLAG_DOCUMENT_2',
        'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
        'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
        'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
        'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

    num_attribs = app_num_attribs + new_app_attribs + prev_aggr_attribs + bureau_aggr_attribs + cash_columns + install_columns + credit_columns

    cat_attribs = app_cat_attribs

    return (processed_data, num_attribs, cat_attribs)


def main():
    processed_data, num_attribs, cat_attribs = load_process_data()
    y = processed_data['TARGET']
    X = processed_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    data_prep_pipeline = make_prep_pipeline(num_attribs, cat_attribs)

    np.random.seed(42)
    full_pipeline_with_predictor = Pipeline([
        ("preparation", data_prep_pipeline),
        # ("feature_selector", SelectFromModel(RandomForestClassifier(
        #     class_weight='balanced',
        #     random_state=0))),
        ("classifier", LogisticRegression(class_weight="balanced"))
    ])

    model = full_pipeline_with_predictor.fit(X_train, y_train)
    print('trained')

    # Evaluating
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    y_train_pred = y_train_pred_proba > 0.5
    y_test_pred = y_test_pred_proba > 0.5
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', verbose=1)
    try:
        expLog
    except NameError:
        expLog = pd.DataFrame(columns=["exp_name",
                                       "Train AUC",
                                       "4-fold Valid AUC",
                                       "4-fold Valid AUC std",
                                       "Test  AUC"
                                       ])

    selected_attribs = num_attribs + cat_attribs
    exp_name = f"Baseline_{len(selected_attribs)}_features"
    expLog.loc[len(expLog)] = [f"{exp_name}"] + list(np.round(
        [roc_auc_score(y_train, y_train_pred_proba),
         scores.mean(),
         scores.std(),
         roc_auc_score(y_test, y_test_pred_proba)],
        4))

    print(expLog.to_string())

    version = datetime.datetime.now().isoformat()
    expLog.to_csv(f'expLog_{version}.csv', index=False)
    # importances.to_csv(f'importances_{version}.csv')
    # dump(lm, f'lm_{version}.joblib')


if __name__ == '__main__':
    main()
