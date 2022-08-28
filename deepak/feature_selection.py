import numpy as np
import pandas as pd
import datetime
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import SCORERS
from sklearn.metrics import roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score

from hcdr_applications import load_datasets, preprocessing_transformations, make_prep_pipeline

# load data
DATA_DIR = "../data"
# ds_names = ("application_train", "application_test", "bureau","bureau_balance","credit_card_balance","installments_payments",
#             "previous_application","POS_CASH_balance")
ds_names = ("application_train", )
datasets = load_datasets(DATA_DIR, ds_names)
print('loaded data')

y = datasets['application_train']['TARGET']
X = preprocessing_transformations(datasets['application_train'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
# X_kaggle_test = datasets['application_test']

data_prep_pipeline, num_attribs_total, cat_attribs = make_prep_pipeline()

full_pipeline_with_predictor = Pipeline([
    ("preparation", data_prep_pipeline),
    ('L1_selector', SelectFromModel(LogisticRegressionCV(
        Cs=np.logspace(-4, -2, 32),
        cv=5,
        penalty='l1',
        solver='saga',
        class_weight='balanced',
        n_jobs=64,
        random_state=0))),
    ("model", LogisticRegression(class_weight='balanced'))
])
model = full_pipeline_with_predictor.fit(X_train, y_train)

print('trained')

# Evaluating
y_train_pred_proba = model.predict_proba(X_train)[:, 1]
y_test_pred_proba = model.predict_proba(X_test)[:, 1]
y_train_pred = y_train_pred_proba > 0.5
y_test_pred = y_test_pred_proba > 0.5
# scores = cross_val_score(model, X_train, y_train, cv=5,
#                          scoring='roc_auc', verbose=1)
try:
    expLog
except NameError:
    expLog = pd.DataFrame(columns=["exp_name",
                                   "Train AUC",
                                   "5-fold Valid AUC",
                                   "5-fold Valid AUC std",
                                   "Test  AUC"
                                   ])

cat_pipeline = data_prep_pipeline.transformer_list[1][1]
cat_features = [f'{base}_{c}'for base, ohe_c in zip(
    cat_attribs, cat_pipeline.named_steps['ohe'].categories_) for c in ohe_c]
features = num_attribs_total + cat_features
total_num_features = len(features)

selector_model = full_pipeline_with_predictor.named_steps['L1_selector']
selected_attribs = list(np.array(features)[selector_model.get_support()])
print(selected_attribs)
print('C', selector_model.estimator_.C_)

exp_name = f"Baseline_{len(selected_attribs)}({len(num_attribs_total + cat_attribs)}.{total_num_features})_features"
expLog.loc[len(expLog)] = [f"{exp_name}"] + list(np.round(
    [roc_auc_score(y_train, y_train_pred_proba),
     0, #scores.mean(),
     0, #scores.std(),
     roc_auc_score(y_test, y_test_pred_proba)],
    4))

# Feature Importances


lm = full_pipeline_with_predictor['model']
importances = pd.DataFrame(lm.coef_.T/np.sum(np.abs(lm.coef_)), index=selected_attribs,
                           columns=['Imp']).abs().sort_values(by='Imp', ascending=False)

# report
print(expLog)
# print(importances)
# print(lm.C_)

version = datetime.datetime.now().isoformat()
expLog.to_csv(f'expLog_{version}.csv', index=False)
importances.to_csv(f'importances_{version}.csv')
# dump(lm, f'lm_{version}.joblib')
dump(selector_model, f'selector_model_{version}.joblib')
