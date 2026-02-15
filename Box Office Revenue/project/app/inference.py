import numpy as np
import pandas as pd
from joblib import load
import pickle
import redis
import hashlib
import json
from app.pipeline import prepare, prepare_single
from app.utils import safe_prepare
import xgboost as xgb

# --- Redis setup with fallback ---
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.ping()  # Test connection
    redis_available = True
except redis.ConnectionError:
    redis_client = None
    redis_available = False

# Load models and blend config at module level (so they're loaded once)
xgb_model = load('models/final_xgb.pkl')
lgb = load('models/final_lgb.pkl')
cat = load('models/final_cat.pkl')
with open('models/blend_config.pkl', 'rb') as f:
    blend = pickle.load(f)
weights = blend["weights"]

# Load preprocess assets
with open("models/train_dict.pkl", "rb") as f:
    train_dict = pickle.load(f)

with open("models/global_stats.pkl", "rb") as f:
    global_stats = pickle.load(f)

with open("models/features.pkl", "rb") as f:
    features = pickle.load(f)


def predict_movie(df, train_dict=train_dict, cache_expire_seconds=36000):
    """
    Prepare the features and predict using the blended ensemble, with Redis caching.
    :param df: DataFrame with input data (should have same columns as in training)
    :param train_dict: dictionary with json/categorical info (from training)
    :param cache_expire_seconds: duration to cache the prediction (in seconds)
    :return: numpy array of predicted revenues (original scale)
    """
    # --- Create a unique cache key for this input ---
    df_json = df.to_json()
    input_hash = hashlib.md5(df_json.encode()).hexdigest()
    cache_key = f"predict_movie:{input_hash}"

    # --- Try Redis cache first (if available) ---
    if redis_available and redis_client:
        try:
            cached_pred = redis_client.get(cache_key)
            if cached_pred:
                print(f"Cache hit for key: {cache_key}")
                return np.array(json.loads(cached_pred))
            else:
                print(f"Cache miss for key: {cache_key}")
        except redis.ConnectionError:
            pass  # Redis not available at runtime, continue without cache

    # --- Feature preparation ---
    df_prep = prepare(df, train_dict)
    df_prep = safe_prepare(df_prep)
    df_prep = df_prep.reindex(columns=features, fill_value=0)

    # --- Predict with each model ---
    # XGBoost model: ensure input is DMatrix if model is core Booster, else use DataFrame for sklearn API
    try:
        pred_xgb = xgb_model.predict(df_prep)
    except TypeError as e:
        # If model requires DMatrix (core XGBoost Booster)
        dmatrix = xgb.DMatrix(df_prep)
        pred_xgb = xgb_model.predict(dmatrix)

    pred_lgb = lgb.predict(df_prep)
    pred_cat = cat.predict(df_prep)
    
    # --- Weighted blending ---
    final_pred = (pred_xgb * weights['xgb'] +
                  pred_lgb * weights['lgb'] +
                  pred_cat * weights['cat'])
    # Reverse the log1p transform (if used during training)
    result = np.expm1(final_pred)

    # --- Cache the result in Redis (if available) ---
    if redis_available and redis_client:
        try:
            redis_client.setex(cache_key, cache_expire_seconds, json.dumps(result.tolist()))
        except redis.ConnectionError:
            pass  # Ignore Redis errors
        
    return result



def single_movie_pred(movie_dict, cache_expire_seconds=36000):
    """
    Prepare features and predict revenue for a single movie dict, with Redis caching.
    :param movie_dict: dict, single movie data (as received from API)
    :param cache_expire_seconds: cache duration in seconds
    :return: float, predicted revenue (original scale)
    """
    # --- Create unique cache key ---
    movie_json = json.dumps(movie_dict, sort_keys=True)
    input_hash = hashlib.md5(movie_json.encode()).hexdigest()
    cache_key = f"single_movie_pred:{input_hash}"

    # --- Try Redis cache first ---
    if redis_available and redis_client:
        try:
            cached_pred = redis_client.get(cache_key)
            if cached_pred:
                print(f"Cache hit for key: {cache_key}")
                return float(json.loads(cached_pred))
            else:
                print(f"Cache miss for key: {cache_key}")
        except redis.ConnectionError:
            pass

    # --- Prepare features ---
    df = pd.DataFrame([movie_dict])
    df_prep = prepare_single(df, train_dict, global_stats, features)
    df_prep = safe_prepare(df_prep)
    df_prep = df_prep.reindex(columns=features, fill_value=0)

    # --- Predict with each model ---
    try:
        pred_xgb = xgb_model.predict(df_prep)
    except TypeError:
        dmatrix = xgb.DMatrix(df_prep)
        pred_xgb = xgb_model.predict(dmatrix)
    pred_lgb = lgb.predict(df_prep)
    pred_cat = cat.predict(df_prep)

    # --- Weighted blending ---
    final_pred = (pred_xgb * weights['xgb'] +
                  pred_lgb * weights['lgb'] +
                  pred_cat * weights['cat'])
    # Reverse log1p if used
    result = np.expm1(final_pred)[0]

    # --- Cache the result ---
    if redis_available and redis_client:
        try:
            redis_client.setex(cache_key, cache_expire_seconds, json.dumps(result))
        except redis.ConnectionError:
            pass

    return float(result)