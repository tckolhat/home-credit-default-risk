from hcdr import make_prep_pipeline, load_process_data
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegressionCV
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


def main():
    processed_data, num_attribs, cat_attribs = load_process_data()
    y = processed_data['TARGET']
    X = processed_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    data_prep_pipeline = make_prep_pipeline(num_attribs, cat_attribs)

    print('preprocessed data')
    np.random.seed(42)
    full_pipeline_with_predictor = Pipeline([
        ("preparation", data_prep_pipeline),
        ("feature_selector", SelectFromModel(LogisticRegressionCV(
            Cs=np.logspace(-3, -1, 2),
            penalty='l1',
            solver='saga',
            cv=5,
            class_weight='balanced',
            n_jobs=5,
            random_state=0))),
        ("classifier", LogisticRegression(class_weight="balanced"))
    ])

    model = full_pipeline_with_predictor.fit(X_train, y_train)
    print('trained')

    cat_pipeline = data_prep_pipeline.transformer_list[1][1]
    cat_features = [f'{base}_{c}'for base, ohe_c in zip(
        cat_attribs, cat_pipeline.named_steps['ohe'].categories_) for c in ohe_c]
    features = num_attribs + cat_features
    print(f'features: {len(features)}, num_attribs: {len(num_attribs)}, cat_features: {len(cat_features)}')

    selector_model = full_pipeline_with_predictor.named_steps['feature_selector']
    selected_features = list(np.array(features)[selector_model.get_support()])
    print(f'attribs: {len(num_attribs + cat_attribs)}, features: {len(features)}, selected_features={len(selected_features)}')

    selected_attribs = set([f if f in num_attribs else '_'.join(f.split('_')[:-1]) for f in selected_features])
    unused_attribs = set(num_attribs+cat_attribs) - selected_attribs

    print('\n\n\nselected')
    print(selected_attribs)

    print('\n\n\nunused')
    print(unused_attribs)

    print('C', selector_model.estimator_.C_)
    print('Fitting again for cross val score')

    full_pipeline_with_predictor = Pipeline([
        ("preparation", data_prep_pipeline),
        ("feature_selector", SelectFromModel(LogisticRegression(
            C=selector_model.estimator_.C_,
            penalty='l1',
            solver='saga',
            class_weight='balanced',
            random_state=0))),
        ("classifier", LogisticRegression(class_weight="balanced"))
    ])

    # Evaluating
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    # y_train_pred = y_train_pred_proba > 0.5
    # y_test_pred = y_test_pred_proba > 0.5
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
