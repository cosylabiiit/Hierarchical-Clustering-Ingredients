import os
import pandas as pd
import numpy as np
import preprocess
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

DB_PR_UT = 'Recipes(6).csv'
DB_ING = 'Recipe_correct_ndb.csv'
DB_NAME = 'data.csv'

def getPartialRecipesSet():
    #create combinations MANUALLY and check....
    pr = preprocess.createSet(df, 'Processes', '||')
    ut = preprocess.createSet(df, 'Utensils', '||')
    # ing = preprocess.createSet(df, 'ingredients')
    #join all
    data=[]
    for i, j in zip(pr,ut):
        m=[]
        preprocess.addtoCommon(i,m)
        preprocess.addtoCommon(j,m)
        # addtoCommon(k,m)

        data.append(m)
    return data

def create_dataset():
    dfall = pd.read_csv(DB_PR_UT, dtype=str)
    dfall2 = pd.read_csv(DB_ING, dtype=str)

    df = preprocess.cleanData(dfall2, dfall)
    data = preprocess.getRecipeSet(df, df, df)

    print(data[:5])

    df['items']=pd.Series(data)

    # print(df['items'].head())
    df.to_csv(DB_NAME)


if os.path.exists(DB_NAME):
    pass
else:
	create_dataset()
    # print(df.head())

df = pd.read_csv(DB_NAME)

# Select Region here
x = list((df['Sub Region'].unique()))
for i in range(len(x)):
    data=df.loc[df['Sub Region']==x[i]]

    #encode the elements of the lists in the items col
    data['items'] = data['items'].str[1:-1]
    data['items'] = data['items'].str.split(',')
    r = data['items'].values.tolist()
    # print(items[:5]
    te = TransactionEncoder()


    te_ary = te.fit(r).transform(r)
    data= pd.DataFrame(te_ary, columns=te.columns_)
    print(len(te.columns_))
    y=fpgrowth(data, min_support=0.2, use_colnames=True)
    print(x[i])
    y.to_csv("./subregion/{}_Pattern".format(str(x[i])))