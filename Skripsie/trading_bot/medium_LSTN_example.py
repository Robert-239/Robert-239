#imports
import math
from keras.api.losses import mean_squared_error
from keras.src.layers.preprocessing.tf_data_layer import keras
import polars as pl
import numpy as np
import time as tm 
import datetime as dt 
from sqlalchemy.sql.expression import False_
import tensorflow as tf 
import tensorboard
print(tensorboard.__version__)  

import matplotlib as mpl

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import deque
import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM,Dense, Dropout ,Activation, Bidirectional, Input
import matplotlib.pyplot as plt


#the window to look at 
WINDOW = 35
RET_WIN = 7

PREDICTED_PRICES = [0 , 0 ,0]

#LOADING DATA 
import polars as pl
import os
from typing import List

#Data Path


FLIST2 = ['C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2019',
          'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2020',
          'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2021',
          'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2022']
#set random seed for reproduce ability
np.random.seed(42)


def load_and_combine_csv(folders: List[str], file_patern: str = '*.csv') -> pl.DataFrame:
    '''

    '''
    combined_df = None
    for folder in folders:
        csv_files = [f for f in os.listdir(folder) if f.endswith(file_patern)]
        for csv_file in csv_files:
            file_path = os.path.join(folder, csv_file)
            df = pl.read_csv(file_path)
            if combined_df is None:
                combined_df = df
            else:
                combined_df = combined_df.extend(df)
    if combined_df is None:
        raise ValueError("No CSV files found in the specified folders.")
    combined_df = combined_df.unique(subset=["timestamp"], maintain_order=True)
    return combined_df

#load data from csv files and format for plotting libraries
#data  = load_and_combine_csv(FLIST2,'_8_hour_price.csv')
data  = pl.read_csv('moving_avarege_8H.csv')
formated_data = pl.DataFrame(schema={
        'timestamp': str,
        'open': pl.Float64,
        'close': pl.Float64,
        'high': pl.Float64,
        'low': pl.Float64,
        'volume': pl.Float64,
    })
#print(data)
lengthDF = data.shape[0]
print(lengthDF)
def ignore():
    for i in range(lengthDF):
        temp_dict = data.row(i,named= True)
        temp_dict['timestamp'] = (dt.datetime.strftime((dt.datetime.strptime(temp_dict['timestamp'],"%Y-%m-%d %H-%M-%S")),"%Y-%m-%d"))
        
        temp_Frame = pl.DataFrame(temp_dict)
        
        formated_data = formated_data.extend(temp_Frame)
    
data_short_mavg = data.drop(['timestamp','long_mavg','rsi'])
data_long_mavg = data.drop(['timestamp','short_mavg','rsi'])
print(data)

scalar = MinMaxScaler(feature_range=(0,1))

#close_price = formated_data.select('close').to_numpy()
#close_price.reshape((close_price.shape[0],1))

short_mavg = data_short_mavg.select('short_mavg').to_numpy()
short_mavg.reshape((short_mavg.shape[0],1))

long_mavg = data_long_mavg.select('long_mavg').to_numpy()
long_mavg.reshape((long_mavg.shape[0],1))

scaled_short_mavg = scalar.fit_transform(short_mavg)
scaled_long_mavg = scalar.fit_transform(long_mavg)

def pre_process(prices : np.ndarray,input_window):
    rng = (prices.shape[0]-input_window)
    data_vec = np.ndarray((rng,input_window))
    target_vec = np.ndarray((rng,1))
    index = 0

    for j in range(prices.shape[0]-(input_window) ):
        
        for i in range(input_window):
            data_vec[j,i] = prices[i+index]
            if i >= input_window - 1:
                target_vec[j,0] = prices[i + 1+ index ]
        index+= 1



    return data_vec , target_vec

data_x_7 , data_y_7 = pre_process(scaled_long_mavg,7)
data_x_14 , data_y_14 = pre_process(scaled_long_mavg,14)
data_x_21 , data_y_21 = pre_process(scaled_long_mavg,21)

print(f"\ndata vectors:\t{data_x_7.shape}")
print(data_x_7)
print(f"\ndata targets:\t{data_y_7.shape}")
print(data_y_7)

print("\nnew window size\n")

# Creating the model using KERAS
print("splitting")
x_train_7 , x_test_7 ,y_train_7, y_test_7 = train_test_split(data_x_7,data_y_7 , test_size= 0.3 , random_state= 42, shuffle= True)
x_train_14 , x_test_14 ,y_train_14, y_test_14 = train_test_split(data_x_14,data_y_14 , test_size= 0.3 , random_state= 42,shuffle= True)
x_train_21 , x_test_21 ,y_train_21, y_test_21 = train_test_split(data_x_21,data_y_21 , test_size= 0.3, random_state= 42, shuffle= True)


print(f"data shape {data_x_7.shape}" )
x_train_7 = np.array(x_train_7)
x_train_7 = np.expand_dims(x_train_7,axis=2)
x_test_7  = np.expand_dims(x_test_7,axis=2)
data_x_7 = np.expand_dims(data_x_7,axis=2)


x_train_14 = np.array(x_train_14)
x_train_14 = np.expand_dims(x_train_14,axis=2)
x_test_14  = np.expand_dims(x_test_14,axis=2)
data_x_14 = np.expand_dims(data_x_14,axis=2)

x_train_21 = np.array(x_train_21)
x_train_21 = np.expand_dims(x_train_21,axis=2)
x_test_21  = np.expand_dims(x_test_21,axis=2)
data_x_21 = np.expand_dims(data_x_21,axis =2)
print(f"split training set = {x_train_7.shape}")

MEM = 35
DROPOUT = 0.3

input = [7,14,21]
ephocs = [ 32 , 56]
train_x = [x_train_7, x_test_14,x_test_21] 
train_y = [y_train_7, y_train_14, y_train_21]

for i in range(len(input)):
    for j in range(len(ephocs)):
        print(train_x[i].shape[-1])
        model = keras.Sequential()
        model.add(Bidirectional(LSTM(input[i],return_sequences=True),input_shape=(input[i], train_x[i].shape[-1]) ))
        model.add(Dropout(rate=DROPOUT))
        model.add(Bidirectional(LSTM(input[i]*2, return_sequences=True)))
        model.add(Dropout(rate=DROPOUT))
        model.add(Bidirectional(LSTM(input[i] * 6, return_sequences=True)))
        model.add(Dropout(rate=DROPOUT))
        model.add(Bidirectional(LSTM(input[i] * 3, return_sequences=True)))
        model.add(Dense(input[i]*2))
        model.add(Dropout(rate=0.2))
        model.add(Bidirectional(LSTM(14, return_sequences=False)))
        model.add(Dense(units= 7))
        model.add(Dense(units=1))
        model.add(Activation("linear"))

        model.compile(loss='mean_absolute_error', optimizer='AdamW', metrics=[ 'mae', 'mse'])

        model.fit(train_x[i],train_y[i], epochs=ephocs[j], shuffle=False, validation_split=0.15)
        model.summary()
        model_name = f"model_MAVG_L_{input[i]}_with_epochs_{ephocs[j]}.keras"
        file_name = f"model_3_{input[i]}_with_epochs_{ephocs[j]}.csv"
        model.save(model_name)
        


#model.evaluate(x_test, y_test)


# Prediction

model = keras.saving.load_model('model_MAVG_L_7_with_epochs_32.keras')


model2 = keras.saving.load_model("model_MAVG_L_7_with_epochs_56.keras") 





y_hat_test = model.predict(x_test_7)
y_hat = model.predict(data_x_7)
y_hat = scalar.inverse_transform(y_hat)
y_hat_test = scalar.inverse_transform(y_hat_test)
model.evaluate(x_test_7,y_test_7)


y_hat_test_2 = model2.predict(x_test_7)
y_hat_2 = model2.predict(data_x_7)
y_hat_2 = scalar.inverse_transform(y_hat_2)
y_hat_test_2 = scalar.inverse_transform(y_hat_test_2)
model2.evaluate(x_test_7,y_test_7)




y_test_test = scalar.inverse_transform(y_test_7)
y_test = scalar.inverse_transform(data_y_7)

#y_test_test = y_test_7
#y_test = data_y_7

print(y_hat.shape)
print(y_hat)

# Plotting
plt.figure(figsize=(18, 9))
plt.plot( y_test, color='green', label='Real moving average')
plt.plot(y_hat, color='orange', label='Predicted long moving average for model with 7 timesteps trained for 32 epochs')
plt.plot(y_hat_2, color='blue', label='Predicted long moving average for model with 7 timesteps trained for 56 epochs')
plt.legend(loc='best')
plt.title('Predicted vs Actual Bitcoin Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.show()



plt.figure(figsize=(18, 9))
plt.plot( y_test_test, color='green', label='Real moving average')
plt.plot(y_hat_test, color='orange', label='Predicted long moving average for model with 7 timesteps trained for 32 epochs')
plt.plot(y_hat_test_2, color='blue', label='Predicted long moving average for model with 7 timesteps trained for 56 epochs')
plt.legend(loc='best')
plt.title('Predicted vs Actual Bitcoin Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.show()


