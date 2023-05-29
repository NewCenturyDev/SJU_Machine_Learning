import math

import matplotlib
import pandas as pd
import datetime as dt

from pytimekr import pytimekr  # 공휴일 체크

matplotlib.use('TkAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.min_rows', 50)
pd.set_option('display.width', 1000)

DATA_IN_PATH = './data/'
PREPED_DATA_OUT_PATH = './data/preped/'
season = {
    1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0
}


def parse_date(d):
    return dt.datetime.strptime(d.split('_')[0], "%Y%m%d")


def load_train_data():
    energy_usage_dataset = pd.read_csv(DATA_IN_PATH + 'Train_dataset.csv')
    return pd.DataFrame({
        'datetime': list(energy_usage_dataset['Time']),
        'weekend': list(map(
            lambda d: parse_date(d).weekday() >= 5 or parse_date(d).strftime("%Y-%m-%d") in pytimekr.holidays(),
            energy_usage_dataset['Time']
        )),
        'vacation': list(map(
            lambda d: parse_date(d).month in [1, 2, 7, 8], energy_usage_dataset['Time']
        )),
        'season': list(map(
            lambda d: season[parse_date(d).month], energy_usage_dataset['Time']
        )),
        'energy_usage': list(map(
            lambda p: float(p), energy_usage_dataset['Power']
        )),
    })


def load_weather_data(data, is_test=False):
    weather_dataset = pd.read_csv(DATA_IN_PATH + 'weather.csv')
    weather_data_processed_df = pd.DataFrame({
        'datetime': list(map(
            lambda tm: tm.split('-')[0] + tm.split('-')[1] + tm.split('-')[2].split(' ')[0] + '_' + tm.split(' ')
            [1].split(':')[0], weather_dataset['tm']
        )),
        'templeture': list(map(
            lambda ta: int(ta), weather_dataset['ta']
        )),
        'windspeed': list(map(
            lambda ws: int(ws), weather_dataset['ws']
        )),
        'humidity': list(map(
            lambda hm: int(hm), weather_dataset['hm']
        )),
        'rainfall': list(map(
            lambda rf: float(0.0 if math.isnan(rf) else rf), weather_dataset['rn']
        )),
    })

    # Attach Weather Data
    attached = pd.merge(
        data, weather_data_processed_df, left_on=['datetime'],
        right_on=['datetime'], how='left'
    )
    if is_test:
        return attached[[
            'datetime', 'weekend', 'vacation', 'season',
            'templeture', 'windspeed', 'humidity', 'rainfall'
        ]]
    else:
        return attached[[
            'datetime', 'weekend', 'vacation', 'season',
            'templeture', 'windspeed', 'humidity', 'rainfall', 'energy_usage'
        ]]


def load_test_data():
    energy_usage_testset = pd.read_csv(DATA_IN_PATH + 'Answer_sheet.csv')
    return pd.DataFrame({
        'datetime': list(energy_usage_testset['Time']),
        'weekend': list(map(
            lambda d: parse_date(d).weekday() >= 5 or parse_date(d).strftime("%Y-%m-%d") in pytimekr.holidays(),
            energy_usage_testset['Time']
        )),
        'vacation': list(map(
            lambda d: parse_date(d).month in [1, 2, 7, 8], energy_usage_testset['Time']
        )),
        'season': list(map(
            lambda d: season[parse_date(d).month], energy_usage_testset['Time']
        )),
    })


# MAIN
train_dataset = load_train_data()
test_dataset = load_test_data()

# 이상치(아웃라이어) 제거 => 0인 데이터들과, 1000 넘는 데이터 (1건) 제거
cleaned_train_dataset = train_dataset.drop(
    train_dataset[train_dataset.energy_usage == 0].index
)
cleaned_train_dataset = cleaned_train_dataset.drop(
    cleaned_train_dataset[cleaned_train_dataset.energy_usage > 1000].index
)

# 1차적으로 단순 기계적 보간 수행
cleaned_train_dataset['datetime_parsed'] = pd.to_datetime(cleaned_train_dataset['datetime'], format='%Y%m%d_%H')
cleaned_train_dataset.set_index('datetime_parsed', drop=True, inplace=True)
cleaned_train_dataset = cleaned_train_dataset.reindex(
    pd.date_range(
        start=cleaned_train_dataset.index.min(), end=cleaned_train_dataset.index.max(), freq='1H'
    )
)

for idx in cleaned_train_dataset.index:
    if pd.isnull(cleaned_train_dataset.at[idx, 'datetime']):
        cleaned_train_dataset.at[idx, 'datetime'] = idx.strftime('%Y%m%d_%H').replace('_0', '_')
        cleaned_train_dataset.at[idx, 'weekend'] = idx.weekday() >= 5 or idx.date() in pytimekr.holidays()
        cleaned_train_dataset.at[idx, 'vacation'] = idx.month in [1, 2, 7, 8]
        cleaned_train_dataset.at[idx, 'season'] = season[idx.month]
cleaned_train_dataset['energy_usage'] = cleaned_train_dataset['energy_usage'].interpolate(method='spline', order=3)
print(cleaned_train_dataset)

# 정제된 데이터 출력
train_with_weather_df = load_weather_data(cleaned_train_dataset, False)
print(train_with_weather_df)
train_with_weather_df.to_csv(PREPED_DATA_OUT_PATH + 'train.csv', index=False)

# 평일/주말 데이터 분류 및 출력
train_with_weekday_df = train_with_weather_df[train_with_weather_df['weekend'] == False]
train_with_weekend_df = train_with_weather_df[train_with_weather_df['weekend'] == True]

train_with_weekday_df.to_csv(PREPED_DATA_OUT_PATH + 'train_with_weekday.csv', index=False)
train_with_weekend_df.to_csv(PREPED_DATA_OUT_PATH + 'train_with_weekend.csv', index=False)

# 테스트 데이터 출력
test_with_weather_df = load_weather_data(test_dataset, True)
test_with_weather_df.to_csv(PREPED_DATA_OUT_PATH + 'test.csv', index=False)
