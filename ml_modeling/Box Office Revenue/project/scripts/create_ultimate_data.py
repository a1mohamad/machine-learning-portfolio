#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
from sqlalchemy import create_engine
import psycopg2


# In[37]:


DB_USER = 'postgres'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'tmdb'
DB_PASS = 'math3141'
engine = create_engine(
            f'postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
)


# In[19]:


df_test = pd.read_csv('./data/test.csv')
df_test_feat = pd.read_csv('./data/TestAdditionalFeatures.csv')
df_train = pd.read_csv('./data/train.csv')
df_train_feat = pd.read_csv('./data/TrainAdditionalFeatures.csv')
df_add_train = pd.read_csv('./data/additionalTrainData.csv')


# In[20]:


df_test = pd.merge(df_test, df_test_feat, how= 'left', on= ['imdb_id'])
df_train = pd.merge(df_train, df_train_feat, how= 'left', on= ['imdb_id'])
df_train = pd.concat([df_train, df_add_train])


# In[25]:


df_train['release_date'] = df_train['release_date'].astype(str).str.replace('-', '/')


# In[31]:


df_train = df_train.to_csv('./data/ultimate_train.csv', index= False)
df_test = df_test.to_csv('./data/ultimate_test.csv', index= False)


# In[38]:


train = pd.read_csv('./data/ultimate_train.csv')
test = pd.read_csv('./data/ultimate_test.csv')


# In[40]:


train_table_name = 'ultimate_train'
test_table_name = 'ultimate_test'
train.to_sql(train_table_name, engine, if_exists= 'replace', index= False)
print(
    f"Data from has been imported into the {train_table_name} table in the {DB_NAME} database."
      )
test.to_sql(test_table_name, engine, if_exists= 'replace', index= False)
print(
    f"Data from has been imported into the {test_table_name} table in the {DB_NAME} database."
      )


# In[ ]:




