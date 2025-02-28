from pandas.core.interchange import dataframe
import pandas as pd
from tf_keras.models import Sequential
from tf_keras.layers import Dense
import tensorflow as tf
import numpy as np
from preprocessing import tokenize


def main(df: dataframe):
    df = tokenize(df, "Weather")
    df = tokenize(df, "Traffic_Level")
    df = tokenize(df, "Time_of_Day")
    df = tokenize(df, "Vehicle_Type")

    df.head()

    train_df = df.sample(frac=0.75, random_state=4)
    val_df = df.drop(train_df.index)

    max_val = train_df.max(axis= 0)
    min_val = train_df.min(axis= 0)
    print(min_val)
    print(max_val)

    range = max_val - min_val
    train_df = (train_df - min_val)/(range)

    val_df =  (val_df- min_val)/range

    X_train = train_df.drop('Delivery_Time_min',axis=1)
    X_val = val_df.drop('Delivery_Time_min',axis=1)
    y_train = train_df['Delivery_Time_min']
    y_val = val_df['Delivery_Time_min']

    input_shape = [X_train.shape[1]]

    input_shape
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1,input_shape=input_shape)])

    model.summary()

    model = tf.keras.Sequential([

        tf.keras.layers.Dense(units=64, activation='relu',
                              input_shape=input_shape),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    model.summary()
    model.compile(optimizer='adam',

                  loss='mae')

    losses = model.fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       batch_size=256,
                       epochs=15,
                       )



    print(df.head())
    print("test")