import datetime as dt
import math

import matplotlib
import pandas as pd
from pytimekr import pytimekr  # 공휴일 체크

matplotlib.use('TkAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 100)
pd.set_option('display.width', 1000)

DATA_IN_PATH = './data/'
PREPED_DATA_OUT_PATH = './data/preped/'


def parse_date(d):
    return dt.datetime.strptime(d.split('_')[0], "%Y%m%d")


def load_train_data():
    energy_usage_dataset = pd.read_csv(DATA_IN_PATH + 'Train_dataset.csv')
    return pd.DataFrame({
        'year': list(map(
            lambda d: int(parse_date(d).year), energy_usage_dataset['Time']
        )),
        'month': list(map(
            lambda d: int(parse_date(d).month), energy_usage_dataset['Time']
        )),
        'date': list(map(
            lambda d: int(parse_date(d).day), energy_usage_dataset['Time']
        )),
        'time': list(map(
            lambda t: int(t.split('_')[1]), energy_usage_dataset['Time']
        )),
        'weekend': list(map(
            lambda d: parse_date(d).weekday() >= 5 or parse_date(d).strftime("%Y-%m-%d") in pytimekr.holidays(),
            energy_usage_dataset['Time']
        )),
        'vacation': list(map(
            lambda d: parse_date(d)
            .month in [1, 2, 7, 8], energy_usage_dataset['Time']
        )),
        'energy_usage': list(map(
            lambda p: float(p), energy_usage_dataset['Power']
        )),
    })


def load_weather_data(data):
    weather_dataset = pd.read_csv(DATA_IN_PATH + 'weather.csv')
    weather_data_processed_df = pd.DataFrame({
        'year': list(map(
            lambda d: int(dt.datetime.strptime(d.split(' ')[0], "%Y-%m-%d").year), weather_dataset['tm']
        )),
        'month': list(map(
            lambda d: int(dt.datetime.strptime(d.split(' ')[0], "%Y-%m-%d").month), weather_dataset['tm']
        )),
        'date': list(map(
            lambda d: int(dt.datetime.strptime(d.split(' ')[0], "%Y-%m-%d").day), weather_dataset['tm']
        )),
        'time': list(map(
            lambda t: int(t.split(' ')[1].split(':')[0]), weather_dataset['tm']
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
    return pd.merge(
        data, weather_data_processed_df, left_on=['year', 'month', 'date', 'time'],
        right_on=['year', 'month', 'date', 'time'], how='left'
    )


def load_test_data():
    energy_usage_testset = pd.read_csv(DATA_IN_PATH + 'Answer_sheet.csv')
    return pd.DataFrame({
        'year': list(map(
            lambda d: int(parse_date(d).year), energy_usage_testset['Time']
        )),
        'month': list(map(
            lambda d: int(parse_date(d).month), energy_usage_testset['Time']
        )),
        'date': list(map(
            lambda d: int(parse_date(d).day), energy_usage_testset['Time']
        )),
        'time': list(map(
            lambda t: int(t.split('_')[1]), energy_usage_testset['Time']
        )),
        'weekend': list(map(
            lambda d: parse_date(d).weekday() >= 5 or parse_date(d).strftime("%Y-%m-%d") in pytimekr.holidays(),
            energy_usage_testset['Time']
        )),
        'vacation': list(map(
            lambda d: parse_date(d)
            .month in [1, 2, 7, 8], energy_usage_testset['Time']
        )),
    })


# MAIN (전처리 수행)

# 원본 데이터 불러오기
train_dataset = load_train_data()

# 이상치(아웃라이어) 제거 => 0인 데이터들과, 1000 넘는 데이터 (1건) 제거
cleaned_train_dataset = train_dataset.drop(
    train_dataset[train_dataset.energy_usage == 0].index
)
cleaned_train_dataset = cleaned_train_dataset.drop(
    cleaned_train_dataset[cleaned_train_dataset.energy_usage > 1000].index
)

train_with_weather_df = load_weather_data(cleaned_train_dataset)
train_with_weather_df.to_csv(PREPED_DATA_OUT_PATH + "train_combined.csv", sep=',', index=False)

test_df = load_test_data()
test_with_weather_df = load_weather_data(test_df)
test_with_weather_df.to_csv(PREPED_DATA_OUT_PATH + "test_combined.csv", sep=',', index=False)
