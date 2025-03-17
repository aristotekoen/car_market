import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import seaborn as sns
import matplotlib.pyplot as plt
from const import *


def predict_price(df, q1, q2, q3):
    price_q1 = q1.predict(df)[0]
    price_q2 = q2.predict(df)[0]
    price_q3 = q3.predict(df)[0]

    if price_q3 <= price_q1:
        return price_q3 , (price_q1 + price_q3)/2, price_q1
    if (price_q1 >= price_q2) or (price_q2 >= price_q3):
        return price_q1,  (price_q3 + price_q1)/2, price_q3


    return price_q1, price_q2, price_q3

def pdp_num(row,X_test_set, feature, models):
    """
    Function to get partial dependency plot for a numerical feature.
    :param row: item to be estimated processed for prediction
    :param X_test_set: test set
    :param feature: name of the feature on which to plot pdp
    :param models: list of q1,q2,q3 models
    :return: plot of the influence of the feature on estimated price for the row in question
    """
    min_val = X_test_set[feature].min()
    max_val = X_test_set[feature].max()
    new_set = X_test_set.copy()
    range_vals = np.linspace(min_val,max_val,20)
    predictions_q1 = []
    predictions_q2 = []
    predictions_q3 = []

    for val in range_vals:
        row[feature] = val
        pred1, pred2, pred3 = predict_price(row, models[0], models[1], models[2])
        predictions_q1.append(pred1)
        predictions_q2.append(pred2)
        predictions_q3.append(pred3)
    plt.plot(range_vals, predictions_q2, label="Estimated price")
    plt.fill_between(range_vals, predictions_q1, predictions_q3,color='blue', alpha=0.3, label='Estimated Price Range')
    plt.title('PDP for feature {}'.format(feature))


def pdp_cat(row, X_test_set, feature, models):
    if feature == 'fuel_type':
        range_vals = fuel_types
    elif feature == 'number_plate_ending':
        range_vals = number_plate_endings
    elif feature == 'drive_type':
        range_vals = drive_types
    elif feature == 'body_type':
        range_vals == body_types
    else:
        range_vals = X_test_set[feature].unique()

    predictions_q1 = []
    predictions_q2 = []
    predictions_q3 = []
    vals = []
    for val in range_vals:
        row[feature] = val
        pred_q1, pred_q2, pred_q3 = predict_price(row, models[0], models[1], models[2])
        predictions_q1.append(pred_q1)
        predictions_q2.append(pred_q2)
        predictions_q3.append(pred_q3)
        vals.append(str(val))
    df = pd.DataFrame({'cats':vals, 'pred_q1': predictions_q1, 'pred_q2':predictions_q2, 'pred_q3':predictions_q3}).sort_values(by='pred_q2', ascending=False)
    plt.scatter(np.arange(df.cats.shape[0]), df.pred_q2)
    plt.fill_between(np.arange(df.cats.shape[0]), df.pred_q1, df.pred_q3, alpha=0.2, color='blue')
    plt.xticks(np.arange(df.cats.shape[0]), df.cats, rotation = 60, fontsize=5)
    plt.title('PDP for feature {}'.format(feature))