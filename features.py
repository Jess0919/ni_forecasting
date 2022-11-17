import pandas as pd
import numpy as np
import lightgbm as lgb
import talib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def make_datetime(data):
    # datetime file
    date = []
    datetime = []
    for index in range(len(data)):
        if data['Tick'] == 1:
            date.append(data['trade_date'].iloc[index])
            datetime.append(data['datetime'].iloc[index])
    Date = pd.DataFrame({'date': date, 'datetime': datetime})
    Date.to_feather('date.feather')


def make_minute(data):
    # sample the index to one record per minute
    tick_features = {}
    tick_features['Close'] = []
    for col in ['price', 'ask', 'ask_qty', 'bid', 'bid_qty', 'open_int', 'volume', 'amount']:
        for f in ['mean', 'std', 'max', 'min']:
            tick_features[col + '_' + f] = []

    index = 0
    last_index = 0
    while index < data.shape[0]:
        if data['Tick'].iloc[index] == 2:
            for col in ['price', 'ask', 'ask_qty', 'bid', 'bid_qty', 'open_int', 'volume', 'amount']:
                tick_features[col + '_mean'].append(data[col][last_index + 1 : index + 1].mean())
                tick_features[col + '_std'].append(data[col][last_index + 1 : index + 1].std())
                tick_features[col + '_max'].append(data[col][last_index + 1 : index + 1].max())
                tick_features[col + '_min'].append(data[col][last_index + 1 : index + 1].min())
                if col == 'price':
                    tick_features['Close'].append(data[col].iloc[index])
            last_index = index
        index = index + 1
    ni_tick_features = pd.DataFrame(tick_features)
    ni_tick_features.to_feather('ni_minute.feather')


def make_diff_period_close_features(data):
    # generate Close-related features with different time scales
    for slot in [1, 2, 5, 10, 30, 60]:
        close_price = []
        high = []
        low = []
        idx = 0
        while idx < len(data):
            bound = np.min([idx+slot, len(data)])
            high.append(data['price_max'][idx : bound].max())
            low.append(data['price_min'][idx : bound].min())
            idx = idx + slot
            if idx < len(data):
                close_price.append(data['Close'].iloc[idx])
            else:
                close_price.append(data['Close'].iloc[len(data) - 1])


        close_price_df = pd.DataFrame({'Close': close_price, 'High': high, 'Low': low})
        # MA 5, 10, 20
        for var in [5, 10, 20]:
            close_price_df['MA' + str(var)] = close_price_df['Close'].rolling(var).mean()
        # RSI 3, 6, 12, 24
        for var in [3, 6, 12, 24]:
            close_price_df['RSI' + str(var)] = talib.RSI(close_price_df['Close'], timeperiod=var)
        # EMA
        close_price_df['EMA12'] = talib.EMA(close_price_df['Close'], timeperiod=12).shift(1)
        close_price_df['EMA26'] = talib.EMA(close_price_df['Close'], timeperiod=26).shift(1)
        # MOMENTUM
        close_price_df['MOM'] = talib.MOM(close_price_df['Close'], timeperiod=5)
        # MACD 1, 2
        close_price_df['DIFF'], close_price_df['DEA'], close_price_df['MACD'] = talib.MACD(close_price_df['Close'],
                                                                                           fastperiod=12, slowperiod=26,
                                                                                           signalperiod=9)
        close_price_df['MACD'] = close_price_df['MACD'] * 2
        # KDJ 5, 9
        close_price_df['talib_K'], close_price_df['talib_D'] = talib.STOCH(close_price_df['High'].values,
                                                   close_price_df['Low'].values,
                                                   close_price_df['Close'].values,
                                                   fastk_period=9,
                                                   slowk_period=5,
                                                   slowk_matype=1,
                                                   slowd_period=5,
                                                   slowd_matype=1)
        close_price_df.loc[:, 'talib_J'] = 3.0 * close_price_df.loc[:, 'talib_K'] - 2.0 * close_price_df.loc[:, 'talib_D']
        close_price_df.drop(columns=['Close', 'High', 'Low'], inplace=True)
        close_price_df = close_price_df.shift(1)
        print(close_price_df.columns)
        features = pd.DataFrame()
        for i in range(len(close_price_df)):
            r = pd.DataFrame(close_price_df.iloc[i]).T
            for i in range(slot):
                features = pd.concat([features, r])
        features.reset_index(inplace=True, drop=True)
        features = features.shift(-1)
        features.to_feather('ni_features_' + str(slot) + '.feather')
        print('finish exporting')


def make_prices_features(data):
    # generate features with different time scales related to price, ask, bid
    for f in ['price', 'ask', 'bid']:
        df = pd.DataFrame()
        for slot in [1, 2, 5, 10, 30, 60]:
            price = []
            #high = []
            #low = []
            idx = 0
            while idx < len(data):
                bound = np.min([idx+slot, len(data)])
                #high.append(data[f + '_max'][idx : bound].max())
                #low.append(data[f + '_min'][idx : bound].min())
                price.append(data[f + '_mean'][idx : bound].mean())
                idx = idx + slot

            price_df = pd.DataFrame({f + '_mean0' + '_m' + str(slot): price})

            for roll in range(1, 21):
                price_df[f + '_mean' + str(roll) + '_m' + str(slot)] = price_df[f + '_mean0' + '_m' + str(slot)].shift(roll)
            for roll in range(20):
                price_df[f + '_rate' + str(roll) + '_m' + str(slot)] = \
                    (price_df[f + '_mean' + str(roll) + '_m' + str(slot)] - price_df[f + '_mean' + str(roll+1) + '_m' + str(slot)]) / price_df[f + '_mean' + str(roll+1) + '_m' + str(slot)]
            #price_df.drop(columns=['High', 'Low'], inplace=True)
            price_df = price_df.shift(1)
            print(price_df.columns)
            features = pd.DataFrame()
            for i in range(len(price_df)):
                r = pd.DataFrame(price_df.iloc[i]).T
                for i in range(slot):
                    features = pd.concat([features, r])
            features.reset_index(inplace=True, drop=True)
            features = features.shift(-1)
            df = pd.concat([df, features], axis=1)
            print('finish' + f + str(slot))
        df.to_feather(f + '_features.feather')
        print('finish ' + f)


def make_qty_features(data):
    # generate features with different time scales related to volume, ask_qty, bid_qty, open_int
    data.rename(columns={'volume_mean': 'volume', 'ask_qty_mean': 'ask_qty', 'bid_qty_mean': 'bid_qty', 'open_int_mean': 'open_int'}, inplace=True)
    print(data.columns)
    for f in ['bid_qty', 'open_int']:
        df = pd.DataFrame()
        for slot in [1, 2, 5, 10, 30, 60, 120, 240, 480]:
            qty = []
            idx = 0
            while idx < len(data):
                bound = np.min([idx + slot, len(data)])
                qty.append(data[f][idx: bound].mean())
                idx = idx + slot

            qty_df = pd.DataFrame({f + '_mean0' + '_m' + str(slot): qty})
            for roll in range(1, 21):
                qty_df[f + '_mean' + str(roll) + '_m' + str(slot)] = qty_df[f + '_mean0' + '_m' + str(slot)].shift(roll)
            for roll in range(20):
                qty_df[f + '_rate' + str(roll) + '_m' + str(slot)] = \
                    (qty_df[f + '_mean' + str(roll) + '_m' + str(slot)] - qty_df[f + '_mean' + str(roll+1) + '_m' + str(slot)]) / qty_df[f + '_mean' + str(roll+1) + '_m' + str(slot)]

            print(qty_df.columns)
            features = pd.DataFrame()
            for i in range(len(qty_df)):
                r = pd.DataFrame(qty_df.iloc[i]).T
                for i in range(slot):
                    features = pd.concat([features, r])
            features.reset_index(inplace=True, drop=True)
            features = features.shift(-1)
            df = pd.concat([df, features], axis=1)
        df.to_feather(f + '_features.feather')
        print('finish ' + f)



def make_labels(data):
    # generate labels at different time scales
    minute_start = data.loc[data['Tick'] == 1]
    minute_end = data.loc[data['Tick'] == 2]
    if len(minute_start) != len(minute_end):
        print(len(minute_start))
        print(len(minute_end))
        exit()
    label = {}
    label['label_long_invest'] = []

    # labels at time scale of 1, 5, 10, 20, 30, 60 minutes
    for n in [1, 5, 10, 20, 30, 60]:
        label['label_long_' + str(n)] = []
        label['label_short_' + str(n)] = []
        label['label_long_rate_' + str(n)] = []
        label['label_short_rate_' + str(n)] = []
        label['label_short_invest' + str(n)] = []

        for i in range(minute_start.shape[0] - 60):
            if n == 1:
                label['label_long_invest'].append(minute_start['ask'].iloc[i + 1])
            label['label_short_invest' + str(n)].append(minute_end['ask'].iloc[i + n])
            # 以收益为label
            label['label_long_' + str(n)].append(minute_end['bid'].iloc[i + n] - minute_start['ask'].iloc[i + 1])
            label['label_short_' + str(n)].append(minute_start['bid'].iloc[i + 1] - minute_end['ask'].iloc[i + n])
            # 以收益率吧为label
            label['label_long_rate_' + str(n)].append(label['label_long_' + str(n)][-1] / minute_start['ask'].iloc[i + 1])
            label['label_short_rate_' + str(n)].append(label['label_short_' + str(n)][-1] / minute_end['ask'].iloc[i + n])

    ni_label = pd.DataFrame(label)
    ni_label.to_feather('ni_labels.feather')


def first_and_last(data):
    # identify the first and last tick in each minute
    tick = []
    datetime = data['datetime']
    i = 0
    while i < len(datetime):
        if i == len(datetime) - 1:
            tick.append(2)
            break
        if i == 0:
            tick.append(1)
            i += 1
            continue
        minute_now = datetime.iloc[i].minute
        minute_next = datetime.iloc[i+1].minute
        if minute_now != minute_next:
            tick.append(2)
            tick.append(1)
            i += 2
        else:
            tick.append(0)
            i += 1
    tick_pd = pd.DataFrame({'Tick': tick})
    data = pd.concat([data, tick_pd], axis=1)
    data.to_csv('ni_with_ticks.csv', index=False)
