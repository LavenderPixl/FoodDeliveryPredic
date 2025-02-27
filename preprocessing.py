import pandas as pd
from pandas import DataFrame

csv_file = "Food_Delivery_Times.csv"


def clean_input() -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df = df.drop(columns=["Order_ID", "Preparation_Time_min"])
    return df


def replace_float(df: DataFrame) -> pd.DataFrame:
    mean = df["Courier_Experience_yrs"].mean()
    mean = round(mean, 0)
    df = df.fillna({"Courier_Experience_yrs": mean})
    return df


def replace_str(df: DataFrame, column: str) -> DataFrame:
    most_frequent = df[column].mode()
    df = df.fillna({column: most_frequent[0]})
    return df


def replace_with_zero(df: DataFrame) -> DataFrame:
    df = df.fillna({"Courier_Experience_yrs": 0})
    return df


def preprocess(replacing: int):
    clean_df = clean_input()
    data = 0
    if replacing == 1 or replacing == 2:
        if replacing == 1:
            # Replaces N/A with mean value.
            mean_data = replace_float(clean_df)
            data = mean_data

        if replacing == 2:
            # Replaces N/A with 0.
            zerod_data = replace_with_zero(clean_df)
            data = zerod_data
    else:
        print("Not correct input.")
        return

    data = replace_str(data, "Weather")
    data = replace_str(data, "Traffic_Level")
    data = replace_str(data, "Time_of_Day")
    print(data)
