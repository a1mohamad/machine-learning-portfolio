import numpy as np
import pandas as pd
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sqlalchemy import create_engine
import psycopg2
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from joblib import dump
from app.utils import get_json_dict, get_dictionary, safe_prepare
from app.pipeline import prepare
from app.train import xgb_model, lgb_model, catboost_model

# ---------- DATABASE CONNECTION ----------
DB_USER = "postgres"
DB_PASSWORD = "math3141"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "tmdb"

# Create SQLAlchemy engine
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ---------- DATA LOADING & PREPARATION ----------
# Load the main training table from PostgreSQL
train = pd.read_sql("SELECT * FROM train_data", engine)
add_train = pd.read_sql("SELECT * FROM additional_train_data", engine)
add_feat_train = pd.read_sql("SELECT * FROM train_additional_features", engine)
# Merge additional features into the main training data:
train = pd.merge(train, add_feat_train, how='left', on=['imdb_id'])
train = pd.concat([train, add_train], ignore_index=True)
# Log-transform revenue target
train['revenue'] = np.log1p(train['revenue'])

# JSON and categorical preprocessing
json_cols = [
    'genres', 'production_companies', 'production_countries',
    'spoken_languages', 'Keywords', 'cast', 'crew'
]
for col in json_cols + ['belongs_to_collection']:
    train[col] = train[col].apply(lambda x: get_dictionary(x) if isinstance(x, str) else x)

train_dict = get_json_dict(train, json_cols)
for col in json_cols:
    remove = [k for k, v in train_dict[col].items() if v < 10 or k == '']
    for k in remove:
        del train_dict[col][k]
with open("models/train_dict.pkl", "wb") as f:
    pickle.dump(train_dict, f)

y = train["revenue"].values
prepared_train = prepare(train, train_dict)
prepared_train = safe_prepare(prepared_train)
X = prepared_train
with open("models/features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# ---------- CROSS-VALIDATION & BLENDING ----------
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)
weights = {'xgb': 0.2, 'lgb': 0.4, 'cat': 0.4}
val_pred = np.zeros(len(X))
cv_score = 0

for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}")

    X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_val = y[trn_idx], y[val_idx]

    # Train each model using your logic from the notebook
    result_xgb = xgb_model(X_train, y_train, X_val, y_val)
    result_lgb = lgb_model(X_train, y_train, X_val, y_val)
    result_cat = catboost_model(X_train, y_train, X_val, y_val)

    # Weighted blend of predictions
    blended_val_pred = (
        result_xgb['val'] * weights['xgb'] +
        result_lgb['val'] * weights['lgb'] +
        result_cat['val'] * weights['cat']
    )
    val_pred[val_idx] = blended_val_pred

    rmse = np.sqrt(mean_squared_error(y_val, blended_val_pred))
    print(f"Blend RMSE: {rmse:.5f}")
    cv_score += rmse / k

print(f"\nAverage 10-Fold CV RMSE: {cv_score:.5f}")

# ---------- FINAL MODEL TRAINING ON ALL DATA ----------
# Train on all data (no validation, use all for training)
final_xgb = xgb_model(X, y, X, y)['model']
final_lgb = lgb_model(X, y, X, y)['model']
final_cat = catboost_model(X, y, X, y)['model']

# Save each model for inference
dump(final_xgb, "models/final_xgb.pkl")
dump(final_lgb, "models/final_lgb.pkl")
dump(final_cat, "models/final_cat.pkl")

# Save the blending weights for inference
blend_config = {
    "weights": weights,
    "models": ["final_xgb.pkl", "final_lgb.pkl", "final_cat.pkl"]
}
with open("models/blend_config.pkl", "wb") as f:
    pickle.dump(blend_config, f)

print("Final models and blend config saved in models/ directory")