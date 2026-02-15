import numpy as np
import json
import math
from datetime import datetime
from sqlalchemy import text

def safe_prepare(df):
    """
    Replace inf/-inf with NaN, fill NaNs with column means, and cast to float32.
    Used for preparing DataFrames with numeric values for ML models.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean())
    return df.astype(np.float32)

def get_dictionary(s):
    """
    Parses a string as JSON, replacing single quotes with double quotes.
    Returns a Python object (list or dict), or an empty list on error.
    Useful for parsing string representations of lists/dicts in DataFrames.
    """
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        return []

def get_json_dict(df, json_columns):
    """
    For each column in json_columns (expected to contain lists of dicts), counts the occurrence of each 'name' field across all rows.
    Returns a dict: {col_name: {name: count, ...}, ...}
    Useful for extracting features from columns containing JSON-like data.
    """
    result = {}
    for col in json_columns:
        counter = {}
        for row in df[col]:
            if not isinstance(row, list):
                continue
            for item in row:
                name = item.get('name')
                if name:
                    counter[name] = counter.get(name, 0) + 1
        result[col] = counter
    return result

def replace_nan_with_none(obj):
    """
    Recursively replaces all NaN floats in a nested object (dict, list, tuple, set) with None.
    Useful for cleaning data before serialization or inference, since NaN is not JSON-serializable.
    """
    if isinstance(obj, float) and math.isnan(obj):
        return None
    elif isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        t = type(obj)
        return t(replace_nan_with_none(v) for v in obj)
    else:
        return obj


def log_prediction(engine, endpoint, input_data, prediction, imdb_id= None, status= "success", error_message= None):
    '''
    
    Store prediction results and metadata into the predictions table.
    
    '''
    with engine.begin() as conn:
        stmt= text("""
            INSERT INTO predictions (timestamp, endpoint, imdb_id, input_data, prediction, status, error_message)
            VALUES (:timestamp, :endpoint, :imdb_id, :input_data, :prediction, :status, :error_message)
        """)
        conn.execute(stmt,{
            "timestamp": datetime.utcnow(),
            "endpoint": endpoint,
            "imdb_id": imdb_id,
            "input_data": json.dumps(input_data),
            "prediction": prediction,
            "status": status,
            "error_message": error_message
        })
    return
