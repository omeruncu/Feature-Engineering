import pandas as pd

from matplotlib import pyplot as plt

from datetime import date

from statsmodels.stats.proportion import proportions_ztest

# Set display options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load():
    data = pd.read_csv("data/titanic.csv")
    return data


#############################################
# Feature Extraction
#############################################

#############################################
# Binary Features: Flag, Bool, True-False
#############################################

df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#############################################
# Deriving Features from Texts
#############################################

df.head()

###################
# Letter Count
###################

df["NEW_NAME_COUNT"] = df["Name"].str.len()

###################
# Word Count
###################

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

###################
# Capturing Special Structures
###################

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean","count"]})

###################
# Generating Variables with Regex
###################

df.head()

df['NEW_TITLE'] = df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

summary = df[["NEW_TITLE", "Survived", "Age"]].groupby("NEW_TITLE").agg({
    "Survived": "mean",
    "Age": ["count", "mean"]
})

summary.columns = ['Survival_Rate', 'Age_Count', 'Age_Mean']
summary = summary.sort_values("Survival_Rate", ascending=False)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(summary.index, summary["Survival_Rate"], color='skyblue', label='Survival Rate')
ax1.set_ylabel("Survival Rate", color='skyblue')
ax1.set_ylim(0, 1)
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.set_title("Unvana Göre Hayatta Kalma Oranı ve Ortalama Yaş")

ax2 = ax1.twinx()
ax2.plot(summary.index, summary["Age_Mean"], color='orange', marker='o', label='Average Age')
ax2.set_ylabel("Average Age", color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#############################################
# Creating Date Variables
#############################################

dff = pd.read_csv("data/course_reviews.csv")
dff.head()
dff.info()

dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d %H:%M:%S")

# year
dff['year'] = dff['Timestamp'].dt.year

# month
dff['month'] = dff['Timestamp'].dt.month

# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff (difference in months between two dates): difference in years + difference in months
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month

# day name
dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head()

# date
#######


#############################################
# Feature Interactions
#############################################
df = load()
df.head()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df.head()

df.groupby("NEW_SEX_CAT")["Survived"].mean()

df.groupby("NEW_AGE_PCLASS")["Survived"].mean()

survival_rates = df.groupby("NEW_AGE_PCLASS")["Survived"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(survival_rates.index.astype(str), survival_rates.values, color='mediumseagreen')
plt.xlabel("NEW_AGE_PCLASS")
plt.ylabel("Survival Rate")
plt.title("NEW_AGE_PCLASS'a Göre Hayatta Kalma Oranı")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
