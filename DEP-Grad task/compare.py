import pandas as pd 


df1 = pd.read_csv("Grad Task - data.csv")
df2 = pd.read_excel("out_final.xlsx")

#print(df1.info())
#print(df2.info())


df3 = df2[df2['balance'] != df1['balance']].copy()

df3 = df3.drop(columns=['tokenID','tokenName','tokenABBV'])
df3.columns = ['index','address' , 'balance']

df3.to_excel("diff_wallets.xlsx")

print(df3.info())
df1.reset_index(drop=True)
df3.reset_index(drop=True)
df1_copy = df1[df1['address'] == df3['address']].copy()
df1_copy = df1_copy.drop(columns=['account_id','asset'])
df3 = df3.join(df1_copy)

print(df3.info())
