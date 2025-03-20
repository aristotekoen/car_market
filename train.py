import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import ShuffleSplit

from const import low_importances


def load_train_test():

    print("Loading training data...")

    client = storage.Client()
    bucket = client.get_bucket('price-estimation')
    blob_train = bucket.get_blob('data/train_test_sets/train_02032024.csv')
    blob_test = bucket.get_blob('data/train_test_sets/test_02032024.csv')

    train_name = "train_02032024.csv"
    test_name = "test_02032024.csv"
    blob_train.download_to_filename(train_name)
    blob_test.download_to_filename(test_name)


    train = pd.read_csv(train_name, index_col=0)
    test = pd.read_csv(test_name, index_col=0)
    return train, test

def save_model(model, model_name):
    path = f"{model_name}.cbm"
    model.save_model(path)
    client = storage.Client()
    bucket = client.get_bucket('price-estimation')
    blob = bucket.blob(f'models/{model_name}.cbm')
    blob.upload_from_filename(path)
    print(f"Model {model_name} saved")

def save_results(df, results_name):
    path = f"{results_name}.csv"
    df.to_csv(path)
    client = storage.Client()
    bucket = client.get_bucket('price-estimation')
    blob = bucket.blob(f'data/results/{results_name}')
    blob.upload_from_filename(path)
    print(f"Results {results_name} saved")

def save_processed_set(df, df_name):
    path = f"{df_name}.csv"
    df.to_csv(path)
    client = storage.Client()
    bucket = client.get_bucket('price-estimation')
    blob = bucket.blob(f'data/train_test_sets/{df_name}')
    blob.upload_from_filename(path)
    print(f"Results {df_name} saved")

def drop_unwanted_columns(train, test):
  print(f"Dropping unwanted columns")
  columns_to_drop = ['uuid', 'label', 'model_stub', 'thumbs', 'price_debatable', 'user_id', 'title',
                     'without_vat', 'seller', 'created', 'seo_json_ld', 'address_long', 'modified',
                     'battery_range', 'variant', 'trim', 'id', 'descriptive_title', 'description']
  train = train.drop(columns=columns_to_drop).copy()
  test = test.drop(columns=columns_to_drop).copy()
  return train, test


def process_types(data_set):
  data_set['fuel_consumption'] = data_set['fuel_consumption'].astype(str).str.replace(',','.').astype(float)
  data_set['battery_charge_time'] = data_set['battery_charge_time'].astype(float)
  data_set['registration_year'] = data_set['registration_year'].astype(float)
  data_set['acceleration'] = data_set['acceleration'].astype(str).str.replace(',','.').astype(float)
  return data_set

def get_train_val_test_pools(train, test, strategy):
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

def detect_outlier_on_groups(df, features, group_cols):
    Q1 = df.groupby(group_cols)[features].quantile(0.25)
    Q3 = df.groupby(group_cols)[features].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df.join(lower_bound, how='left', on=group_cols, rsuffix='_lower').copy()
    df = df.join(upper_bound, how='left', on=group_cols, rsuffix='_upper').copy()

    def detect_outlier(row, feature, lbound, ubound):
        if pd.isna(row[feature]):
            return False
        else:
            return row[feature] < lbound or row[feature] > ubound

    is_outlier = pd.DataFrame(False, index=df.index, columns=features)
    for feat in features:
        is_outlier[feat] = df.apply(
            lambda row: detect_outlier(row, feat, row[f'{feat}_lower'], row[f'{feat}_upper']), axis=1)
    df['is_outlier'] = is_outlier.any(axis=1)
    return df

def compute_lower_bound(series):
    return series.quantile(0.25) - 1.5*(series.quantile(0.75) - series.quantile(0.25))

def remove_outliers(train):
    train['raw_price_log'] = train.raw_price.apply(np.log)
    features_IQR = ['raw_price_log']
    train['level_1'] = train.groupby(by=['brand', 'model', 'registration_year']).transform('count')['raw_price'] > 40
    train['level_2'] = train.groupby(by=['brand', 'model']).transform('count')['raw_price'] > 40
    train['level_3'] = train.groupby(by=['brand']).transform('count')['raw_price'] > 40

    train['is_outlier'] = False
    train[train.level_1] = detect_outlier_on_groups(train[train.level_1], features_IQR, ['brand', 'model', 'registration_year'])
    train[train.level_2 & ~(train.level_1)] = detect_outlier_on_groups(train[train.level_2 & ~(train.level_1)], features_IQR, ['brand', 'model'])
    train[~(train.level_1 | train.level_2) & (train.level_3)] = detect_outlier_on_groups(train[~(train.level_1 | train.level_2) & (train.level_3)], features_IQR, ['brand'])


    engine_size_lower = compute_lower_bound(train['engine_size'])
    engine_power_lower = np.exp(compute_lower_bound(train['engine_power'].apply(lambda x: np.log(x + 1e-10))))

    train['is_engine_outlier'] = (train.engine_power < engine_power_lower) | (train.engine_size < engine_size_lower)
    train['is_outlier_final'] = train.is_outlier | train.is_engine_outlier
    train = train[~train.is_outlier_final].copy()
    train.drop(columns=['raw_price_log','level_1','level_2', 'level_3', 'is_engine_outlier', 'is_outlier',
                        'is_outlier_final'], inplace=True)
    return train

def impute_object_col_on_group(group):
    if group.isna().sum() == group.shape[0]:
        return np.nan
    else:
        return group.mode().iloc[0] if not group.mode().empty else np.nan

def impute_num_on_group(group):
    if group.isna().sum() == group.shape[0]:
        return np.nan
    else:
        return group.mean()

def impute_missing_values_on_feature(df, feature):
    #if object select majority among group, for groups only with nan go one level higher
    df = df.copy()
    df = df.loc[:,['brand','model','registration_year',feature]]

    if (feature in ['seats','doors','number_of_gears','rim_size']) | (df.loc[df[feature].notna(),feature].dtype in ['object', 'bool','int64', 'category']):
        print(f'imputing missing categorical values on {feature}')
        df[f'mode_{feature}'] = df.groupby(['brand','model','registration_year'])[feature].transform(impute_object_col_on_group)
        df.loc[df[feature].isna(),feature] = df.loc[df[feature].isna(),f'mode_{feature}']

        if df[feature].isna().sum() > 0:
            df[f'mode_{feature}'] = df.groupby(['brand','model'])[feature].transform(impute_object_col_on_group)
            df.loc[df[feature].isna(),feature] = df.loc[df[feature].isna(),f'mode_{feature}']

        if df[feature].isna().sum() > 0:
            df[f'mode_{feature}'] = df.groupby(['brand','model'])[feature].transform(impute_object_col_on_group)
            df.loc[df[feature].isna(),feature] = df.loc[df[feature].isna(),f'mode_{feature}']

        if df[feature].isna().sum() > 0:
            df.loc[df[feature].isna(),feature] = df[feature].mode()[0]

        if df[feature].eq(True).any():
            df[feature] = df[feature].map({1.0: True, 0.0: False})
            df[feature] = df[feature].astype(bool)

    else:
        print(f'imputing numerical missing values on {feature}')

        df[f'mean_{feature}'] = df.groupby(['brand','model','registration_year'])[feature].transform(impute_num_on_group)
        df.loc[df[feature].isna(),feature] = df.loc[df[feature].isna(),f'mean_{feature}']

        if df[feature].isna().sum() > 0:
            df[f'mean_{feature}'] = df.groupby(['brand','model'])[feature].transform(impute_num_on_group)
            df.loc[df[feature].isna(),feature] = df.loc[df[feature].isna(),f'mean_{feature}']

        if df[feature].isna().sum() > 0:
            df[f'mean_{feature}'] = df.groupby(['brand','model'])[feature].transform(impute_num_on_group)
            df.loc[df[feature].isna(),feature] = df.loc[df[feature].isna(),f'mean_{feature}']
        if df[feature].isna().sum() > 0:
            df.loc[df[feature].isna(),feature] = df[feature].mean()
    return df[feature]


def preprocess_train_test(train, test, strategy):
    print(f"Preprocessing train test")
    train, test = drop_unwanted_columns(train, test)
    train = process_types(train)
    test = process_types(test)
    if strategy == "all_extras":
        option_columns = [x for x in train.columns if "option" in x]
        train = train.drop(columns=option_columns).copy()
        test = test.drop(columns=option_columns).copy()
        save_processed_set(train, f"processed_train_set_{strategy.upper()}.csv")
        save_processed_set(test, f"processed_test_set_{strategy.upper()}.csv")

    if strategy == "options":
        extra_columns = [x for x in train.columns if "extra" in x]
        train = train.drop(columns=extra_columns).copy()
        test = test.drop(columns=extra_columns).copy()
        save_processed_set(train, f"processed_train_set_{strategy.upper()}.csv")
        save_processed_set(test, f"processed_test_set_{strategy.upper()}.csv")

    if strategy == "remove_outliers":
        rm_columns = [x for x in train.columns if "option" in x]
        train = remove_outliers(train)
        train.drop(columns=rm_columns, inplace=True)
        test.drop(columns=rm_columns, inplace=True)
        save_processed_set(train, f"processed_train_set_{strategy.upper()}.csv")
        save_processed_set(test, f"processed_test_set_{strategy.upper()}.csv")

    if strategy == "drop_extras_and_options":
        rm_columns = [x for x in train.columns if "option" in x or "extra" in x]
        train = train.drop(columns=rm_columns).copy()
        test = test.drop(columns=rm_columns).copy()
        save_processed_set(train, f"processed_train_set_{strategy.upper()}.csv")
        save_processed_set(test, f"processed_test_set_{strategy.upper()}.csv")

    if strategy in "drop_unpractical":
        rm_columns = ['kteo', 'emissions_co2', 'battery_charge_time', 'vehicle_width', 'vehicle_height', 'vehicle_length',
                      'wheelbase', 'torque', 'acceleration', 'number_of_gears', 'gross_weight','top_speed', 'fuel_consumption','registration_month']
        option_columns = [x for x in train.columns if "option" in x]
        rm_columns = rm_columns + option_columns
        train = train.drop(columns=rm_columns).copy()
        test = test.drop(columns=rm_columns).copy()
        save_processed_set(train, f"processed_train_set_{strategy.upper()}.csv")
        save_processed_set(test, f"processed_test_set_{strategy.upper()}.csv")

    if strategy == "drop_low_importance":
        train = train.drop(columns=low_importances).copy()
        test = test.drop(columns=low_importances).copy()
        save_processed_set(train, f"processed_train_set_{strategy.upper()}.csv")
        save_processed_set(test, f"processed_test_set_{strategy.upper()}.csv")

    if strategy == "impute_missing_values":
        rm_columns = ['kteo', 'battery_charge_time'] + [x for x in train.columns if "option" in x]
        train = train.drop(columns=rm_columns).copy()
        test = test.drop(columns=rm_columns).copy()
        nan_cols_train = train.columns[train.isna().any()].tolist()
        nan_cols_test = test.columns[train.isna().any()].tolist()
        for feature in nan_cols_train:
            train[feature] = impute_missing_values_on_feature(train, feature)
        for feature in nan_cols_test:
            test[feature] = impute_missing_values_on_feature(test, feature)
    return train, test


def train_model(train, test, strategy):
    print(f"Training model for strategy {strategy}")
    train_processed, test_processed = preprocess_train_test(train, test, strategy)
    monotonic_constraints = {'is_metallic':1,'crashed':-1, 'is_new':1}
    monotonic_constraints.update({x: 1 for x in train_processed.columns.tolist() if 'extra' in x})

    def objective(trial):
        print('Using common params')
        params = {
            "iterations": trial.suggest_int("iterations", 15000, 15000),
            # We set a high number of iterations as we will let the overfitting detector stop the training
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 10, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "loss_function": trial.suggest_categorical("loss_function", ["RMSE","Quantile:alpha=0.5"]),
            "nan_mode": trial.suggest_categorical('nan_mode', ["Max"]),
            "task_type": trial.suggest_categorical("task_type",["GPU"]),
            "grow_policy": trial.suggest_categorical("grow_policy",["Depthwise","Lossguide"])
        }
        print(f"Launching 4 fold cv for strategy {strategy}")
        # cv_results = cv(train_pool, params = params, fold_count=4, verbose=True, early_stopping_rounds=50)

        kf = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
        cv_results = []
        best_iter = []
        for fold, (train_index, val_index) in enumerate(kf.split(train_processed)):
            print(f"Launching fold {fold}")
            train_fold = train_processed.iloc[train_index]
            val_fold = train_processed.iloc[val_index]

            train_fold_pool, val_fold_pool, y_train_fold, y_val_fold = get_train_val_test_pools(train_fold, val_fold, strategy)
            print(params)
            model = CatBoostRegressor(**params)
            model.fit(train_fold_pool, eval_set=val_fold_pool, early_stopping_rounds=50, verbose=50)
            y_pred = model.predict(val_fold_pool)
            cv_results.append(mean_absolute_error(y_val_fold, y_pred))
            best_iter.append(model.get_best_iteration())
            print(f"Fold {fold} finished")

            trial.report(np.mean(cv_results), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("best_iteration", max(best_iter))
        return np.mean(cv_results)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=30)

    print(f"Best params found")
    best_params = study.best_params
    best_trial = study.best_trial
    best_iteration = best_trial.user_attrs["best_iteration"]
    best_params["iterations"] = best_iteration

    final_model = CatBoostRegressor(**best_params)
    train_pool, test_pool, y_train, y_test = get_train_val_test_pools(train_processed, test_processed,strategy)
    final_model.fit(train_pool, verbose=50)

    y_pred = final_model.predict(test_pool)
    results = pd.DataFrame()
    results['id'] = test.id
    results[f'residuals_{strategy}'] = y_test - y_pred
    results[f'abs_residuals_{strategy}'] = abs(y_test - y_pred)
    results[f'ape_{strategy}'] = abs(y_test - y_pred) / y_test

    print(f"Strategy: {strategy}")
    print(f"MAE: {mean_absolute_percentage_error(y_test, y_pred)}")
    print(f"Saving best model for strategy {strategy}")
    print(f"Saving results for strategy {strategy}")

    save_model(final_model, f"catboost_{strategy}")
    #final_model.save_model(f"/content/drive/MyDrive/car_estimation/best_model_{strategy}.cbm")

    save_results(results, f"results_{strategy}")
    #results.to_csv(f"/content/drive/MyDrive/car_estimation/results_{strategy}.csv", index=False)
    return results, final_model, best_params


def launch_training():
  train, test = load_train_test()
  print(f"Train shape: {train.shape}")
  print(f"Test shape: {test.shape}")
  #train_model(train, test, "remove_outliers")
  #train_model(train, test, "drop_extras_and_options")
  #train_model(train,test,"drop_low_importance")
  train_model(train, test, "all_extras")
  #train_model(train, test, "options")
  #train_model(train,test,"impute_missing_values")

def train_final_models():
    print('Creating training set...')
    train, test = load_train_test()
    #train_all = pd.concat([train, test], axis=0)
    params = {'random_strength': 0.01199857498,
 'verbose': 50,
 'iterations': 11333,
 'nan_mode': 'Max',
 'bagging_temperature': 0.8319957516,
 'grow_policy': 'Depthwise',
 'l2_leaf_reg': 0.6747582331,
 'task_type': 'GPU',
 'depth': 9,
 'min_data_in_leaf': 92,
 'learning_rate': 0.02721930604}
    print(f"Parameters for training will be {params}")

    print("preprocessing training set")
    train_all_processed, test_processed = preprocess_train_test(train, test, "drop_unpractical")

    print("Initialising models")
    model_q1 = CatBoostRegressor(**params, loss_function='Quantile:alpha=0.25')
    model_q2 = CatBoostRegressor(**params, loss_function='Quantile:alpha=0.5')
    model_q3 = CatBoostRegressor(**params, loss_function='Quantile:alpha=0.75')


    print("Getting training pool for training")
    train_pool, _, y_train_all, _ = get_train_val_test_pools(train_all_processed, test_processed, "drop_unpractical")

    print("Training model q1")
    model_q1.fit(train_pool, verbose=50)

    print("Training model q2")
    model_q2.fit(train_pool, verbose=50)

    print("Training model q3")
    model_q3.fit(train_pool, verbose=50)

    print('Saving models')

    save_model(model_q1, "catboost_q1")
    save_model(model_q2, "catboost_q2")
    save_model(model_q3, "catboost_q3")






if __name__ == "__main__":
    train_final_models()