#imports

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from sklearn.preprocessing import StandardScaler

#data frame formatting 

pd.options.display.float_format = '{:20.2f}'.format
pd.set_option('display.max_columns',999)

PATH = 'C:/Users/rober/git/Robert-239/Python/Data_Sci_ProjC:/Users/rober/git/Robert-239/Python/Data_Sci_Proj'
df = pd.read_excel(".\\data\\online_retail_II.xlsx",sheet_name= 0)
print(df.head(10))
