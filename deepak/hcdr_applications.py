from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, f1_score
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import datetime
import warnings
from joblib import dump
warnings.filterwarnings('ignore')


def load_data(in_path, name):
    df = pd.read_csv(in_path)
    return df


def load_datasets(DATA_DIR, ds_names):
    datasets = {}
    for ds_name in ds_names:
        datasets[ds_name] = load_data(os.path.join(
            DATA_DIR, f'{ds_name}.csv'), ds_name)
    return datasets


def pct(x):
    return round(100*x, 3)

# Create a class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


def show_scores(y, y_pred, percentages=False):
    conf_mx = confusion_matrix(y, y_pred)
    if percentages:
        conf_mx = 100*conf_mx/y.shape[0]
    print('scores\n')
    print('precision', precision_score(y, y_pred))
    print('recall   ', recall_score(y, y_pred))
    print('f1       ', f1_score(y, y_pred))
    print('accuracy ', np.sum(y == y_pred)/y.shape[0])

    ax = plt.subplot()
    sns.heatmap(conf_mx, annot=True, fmt='3.1f')

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')


def transform_days(X):
    mask = X > 0
    X[mask] = np.NaN
    # return np.log1p(-1*X)
    return -X


def preprocessing_transformations(df, inplace=False):
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
    df_new['OWN_CAR_AGE'] = SimpleImputer(
        strategy='constant', fill_value=0).fit_transform(df[['OWN_CAR_AGE']]).ravel()
    df['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
    df['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)
    df['CODE_GENDER'].replace('XNA', np.nan, inplace=True)

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
    X['phone_to_employ_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / \
        (1+X['DAYS_EMPLOYED'])
    X['external_source_mean'] = X[['EXT_SOURCE_1',
                                   'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    X['cnt_non_child'] = X['CNT_FAM_MEMBERS'] - X['CNT_CHILDREN']
    X['child_to_non_child_ratio'] = X['CNT_CHILDREN'] / X['cnt_non_child']
    X['income_per_non_child'] = X['AMT_INCOME_TOTAL'] / X['cnt_non_child']
    X['credit_per_person'] = X['AMT_CREDIT'] / X['CNT_FAM_MEMBERS']
    X['credit_per_child'] = X['AMT_CREDIT'] / (1 + X['CNT_CHILDREN'])
    X['credit_per_non_child'] = X['AMT_CREDIT'] / X['cnt_non_child']

    return X


def make_prep_pipeline(num_selected=None, cat_selected=None):
    # Pipeline
    num_attribs = ['AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'AMT_CREDIT', 'REGION_POPULATION_RELATIVE',
                   'DAYS_EMPLOYED', 'DAYS_BIRTH', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_ID_PUBLISH',
                   'DAYS_REGISTRATION', 'DAYS_LAST_PHONE_CHANGE', 'OWN_CAR_AGE', 'OBS_30_CNT_SOCIAL_CIRCLE',
                   'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
                   'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                   'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',
                   'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
                   'HOUR_APPR_PROCESS_START']
    new_features = ['annuity_income_percentage', 'car_to_birth_ratio', 'car_to_employ_ratio', 'children_ratio',
                    'credit_to_annuity_ratio', 'credit_to_goods_ratio', 'credit_to_income_ratio', 'days_employed_percentage',
                    'income_credit_percentage', 'income_per_child', 'income_per_person', 'payment_rate', 'phone_to_birth_ratio',
                    'phone_to_employ_ratio', 'external_source_mean', 'cnt_non_child', 'child_to_non_child_ratio',
                    'income_per_non_child', 'credit_per_person', 'credit_per_child', 'credit_per_non_child']
    num_attribs_total = num_attribs + new_features

    if num_selected is not None:
        num_attribs_total = num_selected

    num_pipeline = Pipeline([
        ('new_features', FunctionTransformer(add_new_features)),
        ('selector', DataFrameSelector(num_attribs_total)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('std_scaler', StandardScaler()),
    ])

    # cat_attribs = ['CODE_GENDER']
    cat_attribs = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
                   'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',
                   'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE',
                   'FLAG_EMAIL', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
                   'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'FLAG_DOCUMENT_2',
                   'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
                   'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
                   'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
                   'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
    if cat_selected is not None:
        cat_attribs = cat_selected

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        #('imputer', SimpleImputer(strategy='most_frequent')),
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(sparse=False, handle_unknown="ignore"))
    ])

    data_prep_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
    ])
    return (data_prep_pipeline, num_attribs_total, cat_attribs)


def main():
    # load data
    DATA_DIR = "../data"
    # ds_names = ("application_train", "application_test", "bureau","bureau_balance","credit_card_balance","installments_payments",
    #             "previous_application","POS_CASH_balance")
    ds_names = ("application_train", )
    datasets = load_datasets(DATA_DIR, ds_names)
    print('loaded data')

    # Preparing data
    y = datasets['application_train']['TARGET']
    X = preprocessing_transformations(datasets['application_train'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    # X_kaggle_test = datasets['application_test']

    # training

    data_prep_pipeline, num_attribs_total, cat_attribs = make_prep_pipeline()

    np.random.seed(42)
    full_pipeline_with_predictor = Pipeline([
        ("preparation", data_prep_pipeline),
        ("linear", LogisticRegression(C=0.005, class_weight='balanced'))
    ])
    model = full_pipeline_with_predictor.fit(X_train, y_train)
    print('trained')

    # Evaluating
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    y_train_pred = y_train_pred_proba > 0.5
    y_test_pred = y_test_pred_proba > 0.5
    scores = cross_val_score(model, X_train, y_train, cv=5,
                             scoring='roc_auc', verbose=1)
    try:
        expLog
    except NameError:
        expLog = pd.DataFrame(columns=["exp_name",
                                       "Train AUC",
                                       "5-fold Valid AUC",
                                       "5-fold Valid AUC std",
                                       "Test  AUC"
                                       ])

    selected_attribs = num_attribs_total + cat_attribs
    exp_name = f"Baseline_{len(selected_attribs)}_features"
    expLog.loc[len(expLog)] = [ f"{exp_name}"] + list(np.round(
        [roc_auc_score(y_train, y_train_pred_proba),
         scores.mean(),
         scores.std(),
         roc_auc_score(y_test, y_test_pred_proba)],
        4))

    # Feature Importances
    cat_pipeline = data_prep_pipeline.transformer_list[1][1]
    cat_features = [f'{base}_{c}'for base, ohe_c in zip(
        cat_attribs, cat_pipeline.named_steps['ohe'].categories_) for c in ohe_c]
    features = num_attribs_total + cat_features
    len(features), len(num_attribs_total), len(cat_features)

    # lm = full_pipeline_with_predictor['linear']
    # importances = pd.DataFrame(lm.coef_.T, index=features, columns=['Name'])

    # report
    print(expLog)
    # print(importances)
    # print(lm.C_)

    version = datetime.datetime.now().isoformat()
    expLog.to_csv(f'expLog_{version}.csv', index=False)
    # importances.to_csv(f'importances_{version}.csv')
    # dump(lm, f'lm_{version}.joblib')


if __name__ == '__main__':
    main()
