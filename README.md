# Home Credit Default Risk Prediction
Primary objective – Determining the repayment ability of the client by employing a stacking ensemble of Logistic Regression, Random Forest, AdaBoost, Gradient Boost classifier models and a MLP model. My team used precision-recall curves, F1 score, AUC-ROC to measure the performance of the models. The project got a Kaggle public score of 0.77558.

## Project Etiquette

- Create a folder called data at the same level as this file and place all csv files there
- Create a folder to put all your work and name it after yourself in lowercase and snakecase.
- Do not commit anything that isn't code.
  - For example, do not commit reports, docs, compilation artifacts, data, Kaggle submission files, editor configs etc. Add the filename wildcards to [.gitignore](.gitignore).
- Maintain a runnable python script that contains your final features, pipeline, and evaluation as you develop code in your notebook. You can use [hcdr-applications.py](deepak/hcdr-applications.py) as a starting point. It reduce the notebook code clutter and helps others to quickly understand what your features and pipeline.

## Gotchas

- Dataset is imbalanced
  - use stratification while splitting
    - `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`
    - cross validation: make sure it is using StratifiedKFold internally
  - specify class weights when training
    - ex: `LogisticRegression(class_weight='balanced'))`

## Features

- [Applications Feature Engineering and Selection](deepak/applications.md)

## Hyperparameter Tuning

- [LogisticRegression](deepak/hyperparameter_tuning.py)
- [KNeighborsClassifier](deepak/hyperparameter_tuning.py)

### Guidelines + Starter Code

Refer to [hyperparameter_tuning.py](deepak/hyperparameter_tuning.py) as a template

- I re-use the transformations, pipeline from [hcdr_applications.py](deepak/hcdr_applications.py).

```python
from hcdr_applications import load_datasets, preprocessing_transformations, make_prep_pipeline

DATA_DIR = "../data"
ds_names = ("application_train", )
datasets = load_datasets(DATA_DIR, ds_names)

y = datasets['application_train']['TARGET']
X = preprocessing_transformations(datasets['application_train'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

data_prep_pipeline, num_attribs_total, cat_attribs = make_prep_pipeline()
```

- I modified the `conductGridSearch` function from AML Homework and use it to tune multiple models.
