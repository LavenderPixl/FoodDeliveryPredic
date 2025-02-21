import math

import pandas as pd
from pandas import DataFrame

csv_file = "Food_Delivery_Times.csv"


def clean_input() -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df = df.drop(columns=["Order_ID", "Preparation_Time_min"])
    print(df)
    return df


def replace_with_mean(df: DataFrame) -> pd.DataFrame:
    mean = df["Courier_Experience_yrs"].mean()
    mean = round(mean, 0)
    df = df.fillna(mean)
    return df


def replace_with_zero(df: DataFrame) -> DataFrame:
    df = df.fillna(0)
    return df


clean_df = clean_input()
mean_data = replace_with_mean(clean_df)
replace_with_zero(clean_df)

