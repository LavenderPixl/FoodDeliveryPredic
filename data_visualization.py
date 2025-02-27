from numpy.f2py.crackfortran import kindselector
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt


def run(df: DataFrame):
    print(f"hello: \n {df.head()}")
    sns.displot(data=df, x=df["Weather"], y=df["Traffic_Level"])
    plt.show()
    sns.boxplot(data=df, x=df["Vehicle_Type"], y=df["Distance_km"])
    plt.show()
    sns.scatterplot(data=df, x=df["Distance_km"], y=df["Delivery_Time_min"], hue="Vehicle_Type")
    plt.show()
