import pandas as pd
import numpy as np
from datetime import datetime
import copy

import matplotlib.pyplot as plt


def chunks_time_Trans(df_buy_rolling):
    buy_volume = df_buy_rolling['Price'].max()
    buy_volume.dropna(inplace=True)
    return buy_volume.index


def chunks_time_Trans_price(df_buy_rolling):
    buy_volume = df_buy_rolling['Price'].max()
    buy_volume.dropna(inplace=True)
    return buy_volume.index, buy_volume.values


def chunks_time_Trans_volume(df_buy_rolling):
    buy_volume = df_buy_rolling['Volume'].max()
    buy_volume.dropna(inplace=True)
    return buy_volume.index, buy_volume.values


def get_original_price(time_freq, df_sample):
    df_sample_time = df_sample.copy()


    df_sample_time['datetime'] = pd.to_datetime(df_sample_time['Time'],  format='%H:%M:%S')

    df_sample_time = df_sample_time.reset_index().set_index('datetime')


    df_buy_grouped = df_sample_time.groupby(pd.Grouper(freq=time_freq))

    date, price = chunks_time_Trans_price(df_buy_grouped)

    results_df = pd.DataFrame(
        {'date': date,
         'rush_order':price
         })

    return results_df


def normalize(data):
    data_min = np.min(data)
    data_max = np.max(data)

    normalized_data = (data - data_min) / (data_max - data_min)

    return normalized_data


def rush_order_Trans(df_buy, time_freq):
    df_buy = df_buy.groupby(df_buy.index).count()
    df_buy[df_buy == 1] = 0
    df_buy[df_buy > 1] = 1
    buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['Volume'].sum()
    buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['Volume'].count()

    buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)

    return buy_volume


def build_features_Trans_S(time_freq, df_sample):
    df_sample_time = df_sample.copy()


    df_sample_time['datetime'] = pd.to_datetime(df_sample_time['Time'],  format='%H:%M:%S')

    df_sample_time = df_sample_time.reset_index().set_index('datetime')

    df_sample_time = df_sample_time[(df_sample_time['Type'] == 'S') ]


    df_buy_grouped = df_sample_time.groupby(pd.Grouper(freq=time_freq))

    date = chunks_time_Trans(df_buy_grouped)

    results_df = pd.DataFrame(
        {'date': date,
         'rush_order_S':rush_order_Trans(df_sample_time, time_freq).values,
         'rush_order':rush_order_Trans(df_sample_time, time_freq).values
         })

    return results_df


def build_features_Trans_B(time_freq, df_sample):
    df_sample_time = df_sample.copy()


    df_sample_time['datetime'] = pd.to_datetime(df_sample_time['Time'],  format='%H:%M:%S')

    df_sample_time = df_sample_time.reset_index().set_index('datetime')

    df_sample_time = df_sample_time[(df_sample_time['Type'] == 'B') ]


    df_buy_grouped = df_sample_time.groupby(pd.Grouper(freq=time_freq))

    date = chunks_time_Trans(df_buy_grouped)

    results_df = pd.DataFrame(
        {'date': date,
         'rush_order_B':rush_order_Trans(df_sample_time, time_freq).values,
         'rush_order':rush_order_Trans(df_sample_time, time_freq).values
         })

    return results_df


def get_OCR(df_sample):
    df_sample_time = df_sample.copy()

    df_sample_time['datetime'] = pd.to_datetime(df_sample_time['Time'],  format='%H:%M:%S')


    sell_side_df = df_sample_time.groupby(['datetime', 'SaleOrderID']).agg({
        'SaleOrderVolume': 'max'
    }).reset_index()

    buy_side_df = df_sample_time.groupby(['datetime', 'BuyOrderID']).agg({
        'BuyOrderVolume': 'max'
    }).reset_index()

    final_agg_df = df_sample_time.groupby('datetime').agg({'Price': 'max'}).reset_index()

    sell_v_maxsum = sell_side_df.groupby('datetime').agg({'SaleOrderVolume':'sum'}).reset_index()
    buy_v_maxsum = buy_side_df.groupby('datetime').agg({'BuyOrderVolume':'sum'}).reset_index()

    final_agg_df['OCR_B'] = buy_v_maxsum['BuyOrderVolume'].diff().abs() / (buy_v_maxsum['BuyOrderVolume'] + sell_v_maxsum['SaleOrderVolume'])


    final_agg_df['OCR_S'] = sell_v_maxsum['SaleOrderVolume'].diff().abs() / (sell_v_maxsum['SaleOrderVolume'] + buy_v_maxsum['BuyOrderVolume'])

    return final_agg_df


def agg_OCR(time_freq, df_sample, df_OCR, BorS = 'B'):
    df_sample_time = df_sample.copy()
    df_sample_time['datetime'] = pd.to_datetime(df_sample_time['Time'],  format='%H:%M:%S')

    df_sample_time = df_sample_time.reset_index().set_index('datetime')

    df_buy_grouped = df_sample_time.groupby(pd.Grouper(freq=time_freq))

    date = chunks_time_Trans(df_buy_grouped)

    print('length of date', len(date))

    results_df_sample = pd.DataFrame(
        {'date': date
        })

    df_OCR_time = df_OCR.copy()

    df_OCR_time = df_OCR_time.reset_index().set_index('datetime')

    date_OCR = df_OCR_time.groupby(pd.Grouper(freq=time_freq))
    date_OCR = date_OCR['Price'].max().index

    print('length of date_OCR', len(date_OCR))

    if BorS == 'B':
        OCR_sum = df_OCR_time.groupby(pd.Grouper(freq=time_freq))['OCR_B'].sum()
    else:
        OCR_sum = df_OCR_time.groupby(pd.Grouper(freq=time_freq))['OCR_S'].sum()

    print('length of OCR_sum', len(OCR_sum))

    results_df_OCR = pd.DataFrame(
        {'date': date_OCR,
         'OCR_sum': OCR_sum.values
        })

    results_df = pd.merge(results_df_sample, results_df_OCR, on='date', how='left')

    return results_df, results_df_OCR, results_df_sample


def merge_B_S(df_rushorder_B, df_rushorder_S):

    df_B = df_rushorder_B.drop(columns=['rush_order'])
    df_S = df_rushorder_S.drop(columns=['rush_order'])

    df_B_indexed = df_B.set_index('date')
    df_S_indexed = df_S.set_index('date')

    merged_df = df_B_indexed.join(df_S_indexed, how='left')

    merged_df['rush_order_S'] = merged_df['rush_order_S'].fillna(0)

    merged_df = merged_df.reset_index()

    merged_df['rush_order'] = merged_df['rush_order_B'] + merged_df['rush_order_S']


    return merged_df


def slice_df_bytime(results_df_original, start_time, end_time):
    start = pd.Timestamp('1900-01-01 ' + start_time)
    end = pd.Timestamp('1900-01-01 ' + end_time)

    results_df = results_df_original[(results_df_original['date'] >= start) & (results_df_original['date'] <= end)]

    results_df = results_df.reset_index(drop=True)

    return results_df


def merge_rush_roc_S(df_rush_ori, df_roc_ori, max_roc=1.2, min_roc=0.8, start_time=None , end_time=None):
    df_rush = df_rush_ori.copy()
    df_roc = df_roc_ori.copy()

    if start_time is not None:
        df_rush = slice_df_bytime(df_rush, start_time, end_time)
        df_roc = slice_df_bytime(df_roc, start_time, end_time)


    max_value = df_roc['OCR_sum'].max()
    min_value = df_roc['OCR_sum'].min()

    df_roc['roc_sum_nor'] = (df_roc['OCR_sum'] - min_value) / (max_value - min_value) * (max_roc - min_roc) + min_roc

    df_rush_indexed = df_rush.set_index('date')
    df_roc_indexed = df_roc.set_index('date')

    merged_df = df_rush_indexed.join(df_roc_indexed, how='left')

    merged_df['roc_sum_nor'] = merged_df['roc_sum_nor'].fillna(1)

    merged_df = merged_df.reset_index()

    merged_df['rush_order_S_new'] = merged_df['rush_order_S'] * merged_df['roc_sum_nor']


    new_S_df = pd.DataFrame({'date': merged_df['date'],
                             'rush_order_S': merged_df['rush_order_S_new'],
                             'rush_order': merged_df['rush_order_S_new']})


    return new_S_df, merged_df


def merge_rush_roc_B(df_rush_ori, df_roc_ori, max_roc=1.2, min_roc=0.8, start_time=None , end_time=None):
    df_rush = df_rush_ori.copy()
    df_roc = df_roc_ori.copy()

    if start_time is not None:
        df_rush = slice_df_bytime(df_rush, start_time, end_time)
        df_roc = slice_df_bytime(df_roc, start_time, end_time)

    max_value = df_roc['OCR_sum'].max()
    min_value = df_roc['OCR_sum'].min()

    df_roc['roc_sum_nor'] = (df_roc['OCR_sum'] - min_value) / (max_value - min_value) * (max_roc - min_roc) + min_roc

    df_rush_indexed = df_rush.set_index('date')
    df_roc_indexed = df_roc.set_index('date')

    merged_df = df_rush_indexed.join(df_roc_indexed, how='left')

    merged_df['roc_sum_nor'] = merged_df['roc_sum_nor'].fillna(1)

    merged_df = merged_df.reset_index()

    merged_df['rush_order_B_new'] = merged_df['rush_order_B'] * merged_df['roc_sum_nor']


    new_B_df = pd.DataFrame({'date': merged_df['date'],
                             'rush_order_B': merged_df['rush_order_B_new'],
                             'rush_order': merged_df['rush_order_B_new']})


    return new_B_df, merged_df
