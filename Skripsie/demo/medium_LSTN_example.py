#imports
import math
from keras.api.losses import mean_squared_error
from keras.src.layers.preprocessing.tf_data_layer import keras
import polars as pl
import numpy as np
import time as tm 
import datetime as dt 
import tensorflow as tf 
import hvplot.polars 

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import deque

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM,Dense, Dropout ,Activation, Bidirectional
import matplotlib.pyplot as plt


#the window to look at 
WINDOW = 7 

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


def load_and_combine_csv(folders: List[str], file_pattern: str = '*.csv') -> pl.DataFrame:
    """
    Load CSV files from specified folders and combine them into a single Polars DataFrame.
    Remove duplicate rows based on all columns.

    :param folders: List of folder paths to search for CSV files.
    :param file_pattern: Pattern for CSV file names. Default is '*.csv'.
    :return: A combined Polars DataFrame with duplicates removed.
    """
    combined_df = None

    for folder in folders:
        # Get all CSV files in the folder matching the pattern
        csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
        
        # Load each CSV file and combine them into a single DataFrame
        for csv_file in csv_files:
            file_path = os.path.join(folder, csv_file)
            df = pl.read_csv(file_path)
            
            
            if combined_df is None:
                combined_df = df
                
            else:
                combined_df = combined_df.extend(df)
                
    # If combined_df is still None, it means no CSV files were found
    if combined_df is None:
        raise ValueError("No CSV files found in the specified folders.")

    # Remove duplicates from the combined DataFrame
    combined_df = combined_df.unique(subset=["timestamp"],maintain_order=True)

    return combined_df

#load data from csv files and format for plotting libraries
data  = load_and_combine_csv(FLIST2)
formated_data = pl.DataFrame(schema={
        'timestamp': str,
        'open': pl.Float64,
        'close': pl.Float64,
        'high': pl.Float64,
        'low': pl.Float64,
        'volume': pl.Float64,
    })
print(data)
lengthDF = data.shape[0]
print(lengthDF)
for i in range(lengthDF):
    temp_dict = data.row(i,named= True)
    temp_dict['timestamp'] = (dt.datetime.strftime((dt.datetime.strptime(temp_dict['timestamp'],"%Y-%m-%d %H-%M-%S")),"%Y-%m-%d"))
    
    temp_Frame = pl.DataFrame(temp_dict)
    
    formated_data = formated_data.extend(temp_Frame)
    
    

print(formated_data)

scalar = MinMaxScaler(feature_range=(0,1))

close_price = formated_data.select('close').to_numpy()
close_price.reshape((close_price.shape[0],1))



scaled_close = scalar.fit_transform(close_price)

def pre_process(prices : np.ndarray,window):
    rng = (prices.shape[0]-window)
    data_vec = np.ndarray((rng,window))
    target_vec = np.ndarray((rng,1))
    index = 0

    for j in range(prices.shape[0]-window):
        
        for i in range(window):
            data_vec[j,i] = prices[i+index]
            if i == window -1:
                target_vec[j] = prices[i + index +1]
        index+= 1



    return data_vec , target_vec

data_x , data_y = pre_process(scaled_close,7)

print(f"\ndata vectors:\t{data_x.shape}")
print(data_x)
print(f"\ndata targets:\t{data_y.shape}")
print(data_y)

# Creating the model using KERAS
print("splitting")
x_train , x_test ,y_train, y_test = train_test_split(data_x,data_y , test_size= 0.3 , random_state= 42)
print(f"data shape {data_x.shape}" )
x_train = np.array(x_train)
x_train = np.expand_dims(x_train,axis=2)
x_test  = np.expand_dims(x_test,axis=2)
print(f"split training set = {x_train.shape}")

MEM = 7
DROPOUT = 0.2

model = keras.Sequential()

model.add(Bidirectional(LSTM(MEM,return_sequences= True),input_shape=(MEM,x_train.shape[-1])))
model.add(Dropout(rate=DROPOUT))
model.add(Bidirectional(LSTM((MEM*2),return_sequences=True)))
model.add(Dropout(rate=DROPOUT))
model.add(Bidirectional(LSTM(MEM,return_sequences=False)))
model.add(Dense(units=1))
model.add(Activation("linear"))

model.compile(loss = 'mean_squared_error',optimizer='adam')

model.fit(x_train,
          y_train,
          epochs= 64,
          shuffle= False,
          validation_split= 0.01)

model.evaluate(x_test,y_test)
model.summary()

y_hat = model.predict(data_x)
y_hat = np.array(y_hat)
print(y_hat.shape)

#y_hat_norm = scalar.inverse_transform(y_hat)
#y_test_norm = scalar.inverse_transform(y_test)

#formated_data.write_csv('formated_test.csv',separator=',')
#plt.plot(np.asarray(formated_data['timestamp'],dtype='datetime64[s]'),formated_data['close'])
plt.plot(y_hat,label = "predicted price",color = 'orange')
plt.plot(data_y,label = "Actual Price",color = "green")
plt.tick_params(axis='x',labelrotation= 90)
plt.legend(loc = 'best')
plt.show()

