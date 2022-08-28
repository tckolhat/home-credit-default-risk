import numpy as np
import pandas as pd
import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import SCORERS
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel

from sklearn.neighbors import KNeighborsClassifier

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

# Speed up computation by selecting important categorical features and excluding others
cat_selected = ['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'FLAG_WORK_PHONE',
                'ORGANIZATION_TYPE', 'FLAG_OWN_CAR', 'OCCUPATION_TYPE', 'REG_CITY_NOT_LIVE_CITY',
                'NAME_FAMILY_STATUS', 'FLAG_PHONE', 'FLAG_OWN_REALTY', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_18',
                'WEEKDAY_APPR_PROCESS_START', 'NAME_TYPE_SUITE', 'FLAG_DOCUMENT_16', 'NAME_HOUSING_TYPE',
                'EMERGENCYSTATE_MODE', 'FLAG_DOCUMENT_3', 'WALLSMATERIAL_MODE']

data_prep_pipeline, num_attribs_total, cat_attribs = make_prep_pipeline(
    cat_selected=cat_selected)


results = pd.DataFrame(columns=[
    "ExpID",
    "Train Score",
    "CV Score",
    "CV Score std",
    "Test Score",
    "Train Time(s)",
    "Test Time(s)"
])


def conductGridSearch(models, params_grid, prep_pipeline, scoring,
                      X_train, y_train, X_test, y_test,
                      results, prefix='', i=0, n_jobs=4, verbose=1):
    # scoring: passed to grid search
    i = 0
    for (name, model) in models:
        i += 1
        # Print model and parameters
        print('****** START', prefix, name, '*****')
        parameters = params_grid.get(name, {})
        print("Parameters:")
        if 'features' in param_grid:
            for p in sorted(params_grid['features']):
                print("\t"+str(p)+": " + str(params_grid['features'][p]))
        for p in sorted(parameters.keys()):
            print("\t"+str(p)+": " + str(parameters[p]))

        # generate the pipeline
        full_pipeline_with_predictor = Pipeline([
            ("preparation", prep_pipeline),
            ('L1_selector', SelectFromModel(LogisticRegression(
                C=0.006404,
                penalty='l1',
                solver='liblinear',
                class_weight='balanced',
                random_state=0))),
            ("predictor", model)
        ])

        # Execute the grid search
        params = params_grid.get('features', {})
        for p in parameters.keys():
            pipe_key = 'predictor__'+str(p)
            params[pipe_key] = parameters[p]
        grid_search = GridSearchCV(full_pipeline_with_predictor, params, scoring=scoring, cv=5,
                                   n_jobs=n_jobs, verbose=verbose)
        grid_search.fit(X_train, y_train)

        # y_train_pred_proba = grid_search.best_estimator_.predict_proba(X_train)[:, 1]
        y_test_pred_proba = grid_search.best_estimator_.predict_proba(X_test)[
            :, 1]
        best_train_score = 1.0 #np.round(roc_auc_score(y_train, y_train_pred_proba), 5)
        best_test_score = np.round(roc_auc_score(y_test, y_test_pred_proba), 5)

        # Best estimator score
        best_cv_score = np.round(grid_search.best_score_, 5)
        print('best_cv_score', best_cv_score)
        best_cv_std = np.round(
            grid_search.cv_results_['std_test_score'][grid_search.best_index_], 5)

        mean_fit_time = np.round(
            grid_search.cv_results_['mean_fit_time'][grid_search.best_index_], 5)
        mean_score_time = np.round(
            grid_search.cv_results_['mean_score_time'][grid_search.best_index_], 5)

        # Collect the best parameters found by the grid search
        print("Best Parameters:")
        best_parameters = grid_search.best_estimator_.get_params()
        param_dump = []
        for param_name in sorted(params.keys()):
            param_dump.append((param_name, best_parameters[param_name]))
            print("\t"+str(param_name)+": " + str(best_parameters[param_name]))
        print("****** FINISH", prefix, name, " *****")
        print("")

        # Record the results
        results.loc[i] = [prefix+name,
                          best_train_score,
                          best_cv_score,
                          best_cv_std,
                          best_test_score,
                          mean_fit_time,
                          mean_score_time]


models = [
    # ('LogisticRegression', LogisticRegression(
    #     solver='liblinear', class_weight='balanced', random_state=0)),
    ('KNeighborsClassifier', KNeighborsClassifier(weights='distance', p=2)),
]

param_grid = {
    'LogisticRegression': {
        'C': np.logspace(-4, 1, 32),
        'penalty': ['l2', 'l1']
    },
    'KNeighborsClassifier': {
        'n_neighbors': [2043, 2243],
        # 'weights': ['uniform', 'distance'],  # distance
        # 'p': [1, 2]  # 2s
    }
}

conductGridSearch(models, param_grid, data_prep_pipeline, 'roc_auc',
                  X_train, y_train, X_test, y_test,
                  results, verbose=3, n_jobs=5)


version = datetime.datetime.now().isoformat()
results.to_csv(f'results_{version}.csv', index=False)

print(results.to_string())
