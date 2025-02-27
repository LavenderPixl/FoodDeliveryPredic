import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from preprocessing import tokenize


def main(df: pd.DataFrame) :
    df = tokenize(df, "Weather")
    df = tokenize(df, "Traffic_Level")
    df = tokenize(df, "Time_of_Day")
    df = tokenize(df, "Vehicle_Type")

    df.head()
    x = df[['Distance_km', 'Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type', 'Courier_Experience_yrs' ]]
    y = df['Delivery_Time_min']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

    mlr = LinearRegression()
    mlr.fit(x_train, y_train)

    # forrest
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(x_train, y_train)
    # forrest

    print("Intercept: ", mlr.intercept_)
    print("Coefficients:")
    list(zip(x, mlr.coef_))

    y_pred_mlr = mlr.predict(x_test)
    print("Prediction for test set: {}".format(y_pred_mlr))
    mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
    mlr_diff.head()

    meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
    meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
    rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
    print('R squared: {:.2f}'.format(mlr.score(x, y) * 100))
    print('Mean Absolute Error:', meanAbErr)
    print('Mean Square Error:', meanSqErr)
    #print('Root Mean Square Error:', rootMeanSqErr)

    # forrest
    # Make predictions
    y_pred = rf_regressor.predict(x_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    meanAbErr = metrics.mean_absolute_error(y_test, y_pred)

    # Sample Prediction
    single_data = x_test.iloc[0].values.reshape(1, -1)
    predicted_value = rf_regressor.predict(single_data)
    #print(f"Predicted Value: {predicted_value[0]}")
    #print(f"Actual Value: {y_test.iloc[0]}")

    # Print results
    print(f"R-squared Score: {r2 * 100 :.2f}")
    print('Mean Absolute Error:', meanAbErr)
    print(f"Mean Squared Error: {mse}")
    # forrest