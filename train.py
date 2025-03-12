import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import os
import optuna
from sklearn.model_selection import KFold, ShuffleSplit

columns_to_drop = [ 'uuid', 'label', 'model_stub', 'thumbs', 'price_debatable', 'user_id', 'title',
                    'without_vat', 'seller', 'created', 'seo_json_ld', 'address_long', 'modified',
                     'battery_range', 'variant', 'trim','id', 'descriptive_title','description']

def load_train_test():
    train = pd.read_csv("data/train_02032024.csv")
    test = pd.read_csv("data/test_02032024.csv")
    return train, test

def drop_unwanted_columns(train, test):
  print(f"Dropping unwanted columns")
  return train.drop(columns=columns_to_drop).copy(), test.drop(columns=columns_to_drop).copy()


def process_types(data_set):
  data_set['fuel_consumption'] = data_set['fuel_consumption'].astype(str).str.replace(',','.').astype(float)
  data_set['battery_charge_time'] = data_set['battery_charge_time']
  data_set['registration_year'] = data_set['registration_year'].astype(float)
  data_set['acceleration'] = data_set['acceleration'].astype(str).str.replace(',','.').astype(float)

def get_train_val_test_pools(train, test):
  print(f"Creating train val test pools")
  print(f"Train shape: {train.shape}")
  print(f"Test shape: {test.shape}")
  X_train, y_train = train.drop(columns=['raw_price']), train['raw_price']
  X_test, y_test = test.drop(columns=['raw_price']), test['raw_price']

  categorical_features = [X_train.columns.get_loc(x) for x in
                          X_train.select_dtypes(include=['object', 'category', 'int', 'bool']).columns.tolist()]

  X_train.iloc[:, categorical_features] = X_train.iloc[:, categorical_features].fillna("nan")
  X_test.iloc[:, categorical_features] = X_test.iloc[:, categorical_features].fillna("nan")
  train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
  test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_features)
  print(f"Train pool shape: {train_pool.shape}")
  print(f"Test pool shape: {test_pool.shape}")
  return train_pool, test_pool, y_train, y_test


def preprocess_train_test(train, test, strategy):
    print(f"Preprocessing train test")
    train, test = drop_unwanted_columns(train, test)
    train = process_types(train)
    test = process_types(test)
    if strategy == "all_extras":
        option_columns = [x for x in train.columns if "option" in x]
        train = train.drop(columns=option_columns).copy()
        test = test.drop(columns=option_columns).copy()

    if strategy == "options":
        extra_columns = [x for x in train.columns if "extra" in x]
        train = train.drop(columns=extra_columns).copy()
        test = test.drop(columns=extra_columns).copy()

    # train_pool, test_pool, y_train, y_test = get_train_val_test_pools(train, test)

    return train, test


from logging import log


def train_model(train, test, strategy):
    print(f"Training model for strategy {strategy}")
    train_processed, test_processed = preprocess_train_test(train, test, strategy)

    def objective(trial):
        params = {
            "iterations": 10000,
            # We set a high number of iterations as we will let the overfitting detector stop the training
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 10, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "loss_function": "Quantile:alpha=0.5",
            "nan_mode": "Max",
            "task_type": "GPU",
            "grow_policy": "Depthwise"
        }
        print(f"Launching 4 fold cv for strategy {strategy}")
        # cv_results = cv(train_pool, params = params, fold_count=4, verbose=True, early_stopping_rounds=50)

        kf = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
        cv_results = []
        for fold, (train_index, val_index) in enumerate(kf.split(train_processed)):
            print(f"Launching fold {fold}")
            train_fold = train_processed.iloc[train_index]
            val_fold = train_processed.iloc[val_index]

            train_fold_pool, val_fold_pool, y_train_fold, y_val_fold = get_train_val_test_pools(train_fold, val_fold)
            print(params)
            model = CatBoostRegressor(**params)
            model.fit(train_fold_pool, eval_set=val_fold_pool, early_stopping_rounds=50, verbose=50)
            y_pred = model.predict(val_fold_pool)
            cv_results.append(mean_absolute_error(y_val_fold, y_pred))
            print(f"Fold {fold} finished")

            trial.report(np.mean(cv_results), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(cv_results)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=30)

    print(f"Best params found")
    best_params = study.best_params
    final_model = CatBoostRegressor(**best_params, task_type="GPU")
    train_pool, test_pool, y_train, y_test = get_train_val_test_pools(train_processed, test_processed)
    final_model.fit(train_pool)

    y_pred = final_model.predict(test_pool)
    results = pd.DataFrame()
    results['id'] = test.id
    results[f'residuals_{strategy}'] = y_test - y_pred
    results[f'abs_residuals_{strategy}'] = abs(y_test - y_pred)
    results[f'ape_{strategy}'] = abs(y_test - y_pred) / y_test

    print(f"Strategy: {strategy}")
    print(f"MSE: {mean_absolute_percentage_error(y_test, y_pred)}")
    print(f"Saving best model for strategy {strategy}")
    print(f"Saving results for strategy {strategy}")
    final_model.save_model(f"/content/drive/MyDrive/car_estimation/best_model_{strategy}.cbm")
    results.to_csv(f"/content/drive/MyDrive/car_estimation/results_{strategy}.csv", index=False)
    return results, final_model, best_params

def launch_training():
  train, test = load_train_test()
  print(f"Train shape: {train.shape}")
  print(f"Test shape: {test.shape}")
  results_all, model_all, best_params_all = train_model(train, test, "all_extras")
  results_options, model_options, best_params_options = train_model(train, test, "options")
  return results_all, model_all, best_params_all, results_options, model_options, best_params_options


if __name__ == "__main__":
    launch_training()