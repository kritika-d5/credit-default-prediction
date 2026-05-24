import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(path):
    df = pd.read_csv(path, index_col=0)
    return df

def clean_data(df):
    #dropping rows where "SeriousDlqin2yrs" is null
    df = df[df["SeriousDlqin2yrs"].notna()]

    #dropping rows where age <= 18
    df = df[df["age"] >= 18]

    #Imputation
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    #Clipping values
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=1.0)

    #Clipping late payment columns

    df["NumberOfTime30-59DaysPastDueNotWorse"] = df["NumberOfTime30-59DaysPastDueNotWorse"].clip(upper=10)
    df["NumberOfTimes90DaysLate"] = df["NumberOfTimes90DaysLate"].clip(upper=10)
    df["NumberOfTime60-89DaysPastDueNotWorse"] = df["NumberOfTime60-89DaysPastDueNotWorse"].clip(upper=10)

    return df

def split_data(df):
    y = df["SeriousDlqin2yrs"]
    X = df.drop(columns=["SeriousDlqin2yrs"])

    # stratify=y ensures the ratio of defaults (0s and 1s) is preserved in both train and test sets
    # important for imbalanced data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    #to handle imbalance
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    return X_train_sm, y_train_sm