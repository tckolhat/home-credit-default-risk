# Application Features - Preprocessing and Engineering

## Numerical (44)

- AMT_INCOME_TOTAL
  - heavily right skewed
- AMT_ANNUITY
  - right skewed; log transform
- AMT_GOODS_PRICE
  - right skewed
- AMT_CREDIT
  - right skewed
- REGION_POPULATION_RELATIVE
- EXT_SOURCE_1
  - 56.38% missing
- EXT_SOURCE_2
  - left skewed
- EXT_SOURCE_3
  - 19.83% missing
- DAYS_EMPLOYED
  - 18.01% missing
  - right skewed (after transforming)
- DAYS_BIRTH
- DAYS_ID_PUBLISH
- DAYS_REGISTRATION
  - right skewed (after transforming)
- DAYS_LAST_PHONE_CHANGE
  - right skewed (after transforming)
- OWN_CAR_AGE
  - 65% missing

one unusual row of id: 272071 where SOCIAL CIRCLE (4 features) values are extremely high

- OBS_30_CNT_SOCIAL_CIRCLE
- DEF_30_CNT_SOCIAL_CIRCLE
- OBS_60_CNT_SOCIAL_CIRCLE
- DEF_60_CNT_SOCIAL_CIRCLE

These set of attributes have 13.5% missing data, I tried removing them but the 5-fold validation AUC on Logistic Regression went down a little bit.

- AMT_REQ_CREDIT_BUREAU_HOUR
- AMT_REQ_CREDIT_BUREAU_DAY
- AMT_REQ_CREDIT_BUREAU_WEEK
- AMT_REQ_CREDIT_BUREAU_MON
- AMT_REQ_CREDIT_BUREAU_QRT
- AMT_REQ_CREDIT_BUREAU_YEAR

Ordinal (processed as numerical, any other way?)

- CNT_CHILDREN
- CNT_FAM_MEMBERS
- REGION_RATING_CLIENT
- REGION_RATING_CLIENT_W_CITY
- HOUR_APPR_PROCESS_START

### Engineered

```python
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
```

## Categorical (47 -> 205)

- NAME_CONTRACT_TYPE
- CODE_GENDER
- FLAG_OWN_CAR
- FLAG_OWN_REALTY
- NAME_TYPE_SUITE
- NAME_INCOME_TYPE
- NAME_EDUCATION_TYPE
- NAME_FAMILY_STATUS
- NAME_HOUSING_TYPE
- OCCUPATION_TYPE
- HOUSETYPE_MODE
- WALLSMATERIAL_MODE
- EMERGENCYSTATE_MODE
- WEEKDAY_APPR_PROCESS_START
- ORGANIZATION_TYPE
- FLAG_MOBIL
- FLAG_EMP_PHONE
- FLAG_WORK_PHONE
- FLAG_CONT_MOBILE
- FLAG_PHONE
- FLAG_EMAIL
- REG_REGION_NOT_LIVE_REGION
- REG_REGION_NOT_WORK_REGION
- LIVE_REGION_NOT_WORK_REGION
- REG_CITY_NOT_LIVE_CITY
- REG_CITY_NOT_WORK_CITY
- LIVE_CITY_NOT_WORK_CITY
- FLAG_DOCUMENT_2
- FLAG_DOCUMENT_3
- FLAG_DOCUMENT_4
- FLAG_DOCUMENT_5
- FLAG_DOCUMENT_6
- FLAG_DOCUMENT_7
- FLAG_DOCUMENT_8
- FLAG_DOCUMENT_9
- FLAG_DOCUMENT_10
- FLAG_DOCUMENT_11
- FLAG_DOCUMENT_12
- FLAG_DOCUMENT_13
- FLAG_DOCUMENT_14
- FLAG_DOCUMENT_15
- FLAG_DOCUMENT_16
- FLAG_DOCUMENT_17
- FLAG_DOCUMENT_18
- FLAG_DOCUMENT_19
- FLAG_DOCUMENT_20
- FLAG_DOCUMENT_21

Features with high missing value % but still used

- WALLSMATERIAL_MODE 50.84%
- HOUSETYPE_MODE 50.18%
- EMERGENCYSTATE_MODE 47.40%
- OCCUPATION_TYPE 31.35%

## Transformations

function: `preprocessing_transformations`

- `AMT_ANNUITY`: log transform
- `['DAYS_EMPLOYED', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_LAST_PHONE_CHANGE']`: remove positive values and multiply by -1
- `OWN_CAR_AGE`: replace missing values with 0

## Preprocessing Pipeline

- Numerical Attrs:
  - replace missing values with mean
  - standardize
- Categorical
  - replace missing values with `'missing'`
  - One Hot Encode

## Feature Selection

Eliminate features using L1 normalized Logistic Regression. Total number of features reduced to 77 from 250.

see:

- [feature_selection.py](feature_selection.py)
- [feature_selection.ipynb](feature_selection.ipynb)

34 out of 50 numerical features are used:

- AMT_ANNUITY
- AMT_GOODS_PRICE
- AMT_REQ_CREDIT_BUREAU_DAY
- AMT_REQ_CREDIT_BUREAU_MON
- AMT_REQ_CREDIT_BUREAU_QRT
- AMT_REQ_CREDIT_BUREAU_WEEK
- AMT_REQ_CREDIT_BUREAU_YEAR
- DAYS_BIRTH
- DAYS_EMPLOYED
- DAYS_ID_PUBLISH
- DAYS_REGISTRATION
- DEF_30_CNT_SOCIAL_CIRCLE
- DEF_60_CNT_SOCIAL_CIRCLE
- EXT_SOURCE_1
- EXT_SOURCE_2
- EXT_SOURCE_3
- HOUR_APPR_PROCESS_START
- OBS_30_CNT_SOCIAL_CIRCLE
- OWN_CAR_AGE
- REGION_POPULATION_RELATIVE
- REGION_RATING_CLIENT_W_CITY
- annuity_income_percentage
- car_to_birth_ratio
- child_to_non_child_ratio
- children_ratio
- credit_per_person
- credit_to_goods_ratio
- credit_to_income_ratio
- days_employed_percentage
- external_source_mean
- income_per_child
- payment_rate
- phone_to_birth_ratio
- phone_to_employ_ratio

20 out of 47 categorical features are used:

- CODE_GENDER
- EMERGENCYSTATE_MODE
- FLAG_DOCUMENT_16
- FLAG_DOCUMENT_18
- FLAG_DOCUMENT_3
- FLAG_DOCUMENT_6
- FLAG_OWN_CAR
- FLAG_OWN_REALTY
- FLAG_PHONE
- FLAG_WORK_PHONE
- NAME_EDUCATION_TYPE
- NAME_FAMILY_STATUS
- NAME_HOUSING_TYPE
- NAME_INCOME_TYPE
- NAME_TYPE_SUITE
- OCCUPATION_TYPE
- ORGANIZATION_TYPE
- REG_CITY_NOT_LIVE_CITY
- WALLSMATERIAL_MODE
- WEEKDAY_APPR_PROCESS_START
