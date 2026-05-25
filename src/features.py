def engineer_features(df):
    df = df.copy()
    df["TotalDelinquencies"] = df["NumberOfTime30-59DaysPastDueNotWorse"] + df["NumberOfTime60-89DaysPastDueNotWorse"] + df["NumberOfTimes90DaysLate"]
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["DebtToIncome"] = df["DebtRatio"] / (df["MonthlyIncome"] + 1)
    return df