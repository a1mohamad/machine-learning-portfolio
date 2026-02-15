import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor as cb

def xgb_model(train_x, train_y, val_x, val_y, random_seed=42, verbose=False):
    params = {
        'objective': 'reg:squarederror',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.6,
        'colsample_bytree': 0.7,
        'eval_metric': 'rmse',
        'seed': random_seed,
        'tree_method': 'hist'
    }
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dval = xgb.DMatrix(val_x, label=val_y)
    evals_result = {}
    model = xgb.train(
        params,
        dtrain=dtrain,
        num_boost_round=100000,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=2500,
        verbose_eval=verbose,
        evals_result=evals_result
    )
    best_idx = model.best_iteration
    val_pred = model.predict(dval, iteration_range=(0, best_idx + 1))
    importance = list(model.get_score(importance_type='weight').values())
    return {
        'model': model,
        'val': val_pred,
        'error': evals_result['valid']['rmse'][best_idx],
        'importance': importance
    }

def lgb_model(train_x, train_y, val_x, val_y, random_seed=42, verbose=False):
    params = {
        'objective': 'regression',
        'num_leaves': 30,
        'min_data_in_leaf': 20,
        'max_depth': 9,
        'learning_rate': 0.005,
        'feature_fraction': 0.9,
        'bagging_freq': 1,
        'bagging_fraction': 0.9,
        'lambda_l1': 0.2,
        'bagging_seed': random_seed,
        'metric': 'rmse',
        'random_state': random_seed,
        'verbosity': -1
    }
    train_set = lgb.Dataset(train_x, label=train_y)
    val_set = lgb.Dataset(val_x, label=val_y)
    record = dict()
    model = lgb.train(
        params,
        train_set=train_set,
        num_boost_round=10000,
        valid_sets=[val_set],
        valid_names=['valid_0'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=500),
            lgb.record_evaluation(record),
            lgb.log_evaluation(period=verbose),
        ]
    )
    best_idx = np.argmin(np.array(record['valid_0']['rmse']))
    val_pred = model.predict(val_x, num_iteration=model.best_iteration)
    return {
        'model': model,
        'val': val_pred,
        'error': record['valid_0']['rmse'][best_idx],
        'importance': model.feature_importance('gain'),
    }

def catboost_model(train_x, train_y, val_x, val_y, random_seed=42, verbose=False):
    model = cb(
        iterations=100000,
        learning_rate=0.005,
        depth=5,
        colsample_bylevel=0.8,
        eval_metric='RMSE',
        metric_period=None,
        random_seed=random_seed,
        bagging_temperature=0.2,
        early_stopping_rounds=500,
        verbose=verbose
    )
    model.fit(train_x, train_y,
              eval_set=(val_x, val_y),
              use_best_model=True,
              verbose=verbose)
    val_pred = model.predict(val_x)
    return {
        'model': model,
        'val': val_pred,
        'error': model.get_best_score()['validation']['RMSE'],
        'importance': model.get_feature_importance()
    }