import pandas as pd
import numpy as np

def analyze_distribution(df, quantiles=4):
    df["quantile"] = pd.qcut(df["value"], q=quantiles, labels=[f"Q{i+1}" for i in range(quantiles)])
    
    # Підрахунок кількості значень у кожному квантилі
    distribution = df["quantile"].value_counts().sort_index()
    
    print("Розподіл значень по квантилях:")
    print(distribution)
    
    # Додатково можна порахувати середнє значення у кожному квантилі
    mean_values = df.groupby("quantile", observed=False)["value"].mean()
    print("\nСереднє значення у кожному квантилі:")
    print(mean_values)
    
    return df

if __name__ == "__main__":
    df = pd.read_parquet("large_dataset.parquet")
    df = analyze_distribution(df, quantiles=4)
