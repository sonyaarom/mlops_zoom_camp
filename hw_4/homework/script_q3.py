import pickle
import pandas as pd
import numpy as np
import os

def read_data(filename):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

df = read_data('data/yellow_tripdata_2023-03.parquet')

categorical = ['PULocationID', 'DOLocationID']
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)

y_pred = model.predict(X_val)

df['pred'] = y_pred
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['year'] = df['tpep_pickup_datetime'].dt.year
df['month'] = df['tpep_pickup_datetime'].dt.month
df['ride_id'] = df.apply(lambda row: f'{row["year"]:04d}/{row["month"]:02d}_{row.name}', axis=1)

df_result = pd.DataFrame({
    'ride_id': df['ride_id'],
    'pred': df['pred']
})

df_result.to_parquet(
    'df_result.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)
