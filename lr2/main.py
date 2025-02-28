import pandas as pd
import numpy as np
import dask.dataframe as dd
from joblib import Memory
import functools
import sqlite3


# 1. Генерація великого набору даних (10 млн рядків)
def generate_large_dataset():
    num_rows = 10_000_000
    df = pd.DataFrame({
        "id": np.arange(num_rows),
        "category": np.random.choice(["A", "B", "C", "D"], num_rows),
        "value": np.random.uniform(1, 1000, num_rows),
        "date": pd.date_range("2020-01-01", periods=num_rows, freq="T")
    })
    df.to_csv("large_dataset.csv", index=False)
    df.to_parquet("large_dataset.parquet")
    return df


# 2. Читання великих файлів з chunksize
def read_large_csv():
    chunks = pd.read_csv("large_dataset.csv", chunksize=100_000)
    for chunk in chunks:
        print(chunk.head())
        break  # Тільки перші 5 рядків першого чанка


# 3. Оптимізація DataFrame: категоріальні та datetime дані


def optimize_dataframe(df):
    df["category"] = df["category"].astype("category")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    print(df.info(memory_usage="deep"))


# 4. Кешування результатів
memory = Memory("./cachedir", verbose=0)


@memory.cache
def expensive_computation(df):
    return df.groupby("category")["value"].mean()


# 5. Робота з MultiIndex
def multiindex_demo(df):
    df.set_index(["date", "category"], inplace=True)
    print(df.xs("A", level="category").head())


# 6. Об'єднання великих CSV файлів
def merge_large_csv():
    df1 = pd.read_csv("large_dataset.csv", nrows=500_000)
    df2 = pd.read_csv("large_dataset.csv", skiprows=500_000, nrows=500_000)
    merged_df = pd.concat([df1, df2])
    print(merged_df.info())


# 7. Конвертація в SQLite
def save_to_sql(df):
    conn = sqlite3.connect("large_data.db")
    df.to_sql("data", conn, if_exists="replace", index=False)
    conn.close()


# 8. Оптимізація .apply() за допомогою numpy
def slow_function(x):
    return x ** 2 - np.log(x + 1)


def optimize_apply(df):
    df["computed"] = df["value"].apply(slow_function)  # Повільно
    vectorized_func = np.vectorize(slow_function)
    df["computed_fast"] = vectorized_func(df["value"])  # Швидше
    print(df.head())


if __name__ == "__main__":
    df = generate_large_dataset()
    read_large_csv()
    optimize_dataframe(df)
    print(expensive_computation(df))
    multiindex_demo(df)
    merge_large_csv()
    save_to_sql(df)
    optimize_apply(df)