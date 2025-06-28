import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

import missingno as msno

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load():
    data = pd.read_csv("data/titanic.csv")
    return data


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Gives the names of categorical, numeric and categorical but cardinal variables in the data set.
    Note: Categorical variables with numeric appearance are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
            Dataframe from which variable names are to be taken
        cat_th: int, optional
            Class threshold value for numeric but categorical variables
        car_th: int, optional
            Class threshold value for categorical but cardinal variables    

    Returns
    ------
        cat_cols: list
            Categorical variable list
        num_cols: list
            Numerical variable list
        cat_but_car: list
            Cardinal variable list with categorical view

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The total of the 3 lists that are returned is equal to the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


#############################################
# Missing Values 
#############################################

#############################################
# Capturing Missing Values
#############################################

df = load()
df.head()

# whether there are any missing observations
df.isnull().values.any()

# missing value counts for each variable
df.isnull().sum()

# counts of non-missing values for each variable
df.notnull().sum()

# total number of missing values in the dataset
df.isnull().sum().sum()

# observation units with at least one missing value
df[df.isnull().any(axis=1)]

# observation units with no missing values
df[df.notnull().all(axis=1)]

# Sort in descending order
df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

missing_values_table(df, True)


#############################################
# Solving the Missing Value Problem
#############################################

###################
# Solution 1: Quickly Remove
###################
df.dropna().shape

###################
# Solution 2: Fill with Simple Imputation Methods
###################

df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

# df.apply(lambda x: x.fillna(x.mean()), axis=0)

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

df["Embarked"].fillna("missing")

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

###################
# Assigning Values in Categorical Variable Breakdown
###################


df.groupby("Sex")["Age"].mean()

df["Age"].mean()

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

#############################################
# Solution 3: Fill with Predictive Imputation
#############################################

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

# standardization of variables
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


# kNN imputation
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]


###################
# Recap
###################

df = load()
# missing table
missing_values_table(df)
# fill numeric variables with median
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# fill categorical variables with mode
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# fill numeric variables in categorical breakdown
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# fill with Predictive Imputation
#####


#############################################
# Advanced Analytics
#############################################

###################
# Examining the Structure of Missing Data
###################

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

###################
# Examining the Relationship Between Missing Values and the Target Variable
###################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Survived", na_cols)



###################
# Recap
###################

df = load()
na_cols = missing_values_table(df, True)
# fill numeric variables with median
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# fill categorical variables with mode
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# fill numeric variables in categorical breakdown
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Fill with Predictive Imputation
missing_vs_target(df, "Survived", na_cols)