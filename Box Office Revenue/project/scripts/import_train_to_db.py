from sqlalchemy import create_engine
import psycopg2
import pandas as pd

# Configuration for the database connection
DB_USER = 'postgres'
DB_PASSWORD = 'math3141'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'tmdb'

train_path = 'data/train.csv'
add_train_path = 'data/AdditionalTrainData.csv'
train_table_name = 'train_data'
add_train_table_name = 'additional_train_data'

# Create a database engine
engine = create_engine(
    f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
        )

# Read the CSV files into a DataFrame
train = pd.read_csv(train_path)
add_train = pd.read_csv(add_train_path)
add_train['release_date'] = add_train['release_date'].astype(str).str.replace('-', '/')

# Write the DataFrame to the PostgreSQL table
train.to_sql(train_table_name, engine, if_exists='replace', index=False)
print(
    f"Data from {train_path} has been imported into the {train_table_name} table in the {DB_NAME} database."
      )
add_train.to_sql(add_train_table_name, engine, if_exists='replace', index=False)
print(
    f"Data from {add_train_path} has been imported into the {add_train_table_name} table in the {DB_NAME} database."
      )
# Additional features can be added similarly
feat_path = 'data/TrainAdditionalFeatures.csv' 
feat_table_name = 'train_additional_features'

# Read the additional features CSV file into a DataFrame
add_feat = pd.read_csv(feat_path)
# Write the DataFrame to the PostgreSQL table  
add_feat.to_sql(feat_table_name, engine, if_exists='replace', index=False)

print(f"Data from {feat_path} has been imported into the {feat_table_name} table in the {DB_NAME} database.")
# Close the engine connection