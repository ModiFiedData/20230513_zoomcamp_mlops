#!/usr/bin/env python
# coding: utf-8
import pickle
import pandas as pd
import sys

categorical = ["PULocationID", "DOLocationID"]

def load_model():
    with open("model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def read_data(filename):

    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def run():
    print(f'reading data for {year:04d}-{month:02d}...')
    df = read_data(
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    )

    print('making predictions ...')
    dicts = df[categorical].to_dict(orient="records")
    dv, model = load_model()
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)


    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")


    df_result = pd.DataFrame()
    df_result["ride_id"] = df.ride_id.values
    df_result["predictions"] = y_pred

    print(f"Mean prediction : {df_result.predictions.mean()}")
    
    print('saving output ...')
    output_file = f"./output/df_result{year:04d}-{month:02d}.parquet"
    
    df_result.to_parquet(output_file, engine="pyarrow", compression=None, index=False)
    print('done.')

if __name__ == "__main__":

    year = int(sys.argv[1])
    month = int(sys.argv[2] )
    run()