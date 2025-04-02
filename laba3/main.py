import os
import time
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask_ml.cluster import KMeans as DaskKMeans
import vaex
from sklearn.cluster import MiniBatchKMeans  # Оптимізований KMeans для Vaex

# 1. Генерація великого набору даних
def generate_energy_dataset():
    file_path = "energy_dataset.parquet"
    
    if os.path.exists(file_path):
        print("Файл уже існує. Генерація пропускається.")
        return
    
    print("Генеруємо новий набір даних...")
    num_rows = 10_000_000
    df = pd.DataFrame({
        "id": np.arange(num_rows),
        "region": np.random.choice(["North", "South", "East", "West"], num_rows),
        "consumption_kwh": np.random.uniform(100, 5000, num_rows),
        "temperature_C": np.random.uniform(-10, 40, num_rows),
        "humidity_%": np.random.uniform(20, 90, num_rows),
        "timestamp": np.random.choice(pd.date_range("2023-01-01", periods=100_000, freq="T"), num_rows)
    })
    
    df.to_parquet(file_path)
    print("Файл успішно збережено!")

# 2. Кластеризація з Dask
def cluster_with_dask():
    file_path = "energy_dataset.parquet"
    
    if not os.path.exists(file_path):
        print("Помилка: Файл не знайдено! Спочатку згенеруйте дані.")
        return None

    print("Читаємо дані з файлу за допомогою Dask...")
    ddf = dd.read_parquet(file_path)

    features = ddf[["consumption_kwh", "temperature_C", "humidity_%"]].astype("float64")

    cluster = LocalCluster()
    client = Client(cluster)

    print("Виконуємо кластеризацію з Dask...")
    start_time = time.time()

    kmeans = DaskKMeans(n_clusters=4)
    ddf["cluster"] = kmeans.fit_predict(features)

    dask_time = time.time() - start_time
    print(f"Кластеризація Dask завершена за {dask_time:.2f} сек.")
    
    return ddf, dask_time

# 3. Кластеризація з Vaex
def cluster_with_vaex():
    file_path = "energy_dataset.parquet"
    
    if not os.path.exists(file_path):
        print("Помилка: Файл не знайдено! Спочатку згенеруйте дані.")
        return None

    print("Читаємо дані з файлу за допомогою Vaex...")
    vdf = vaex.open(file_path)

    features = vdf[["consumption_kwh", "temperature_C", "humidity_%"]].to_pandas_df().values

    print("Виконуємо кластеризацію з Vaex...")
    start_time = time.time()

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=10000)
    vdf["cluster"] = kmeans.fit_predict(features)

    vaex_time = time.time() - start_time
    print(f"Кластеризація Vaex завершена за {vaex_time:.2f} сек.")
    
    return vdf, vaex_time

# Запуск та порівняння продуктивності
if __name__ == "__main__":
    generate_energy_dataset()

    print("\n--- Кластеризація з Dask ---")
    ddf, dask_time = cluster_with_dask()

    print("\n--- Кластеризація з Vaex ---")
    vdf, vaex_time = cluster_with_vaex()

    print("\n=== Порівняння часу виконання ===")
    print(f"Dask: {dask_time:.2f} сек.")
    print(f"Vaex: {vaex_time:.2f} сек.")
