import pandas as pd
import matplotlib.pyplot as plt

files = {
    "10 см": "data_10cm.csv",
    "50 см": "data_50cm.csv",
    "100 см": "data_100cm.csv"
}

average_fields = {}
distance_cm = []
all_data = {}

print("Статистика:\n")
for label, file_path in files.items():
    df = pd.read_csv(file_path)
    abs_field = df["Absolute field (µT)"]
    average_fields[label] = abs_field.mean()
    distance = int(label.split()[0])
    distance_cm.append(distance)
    all_data[distance] = df

    print(f"{label}")
    print(f"Середнє: {abs_field.mean():.2f} µT")
    print(f"Стандартне відхилення: {abs_field.std():.2f} µT")
    print(f"Максимум: {abs_field.max():.2f} µT")
    print(f"Мінімум: {abs_field.min():.2f} µT\n")

# === Побудова графіка "Магнітне поле vs Відстань" ===
sorted_distances = sorted(average_fields.items(), key=lambda x: int(x[0].split()[0]))
distances = [int(label.split()[0]) for label, _ in sorted_distances]
field_values = [value for _, value in sorted_distances]

plt.figure(figsize=(8, 5))
plt.plot(distances, field_values, marker='o', linestyle='-', color='blue')
plt.title("Залежність рівня магнітного поля від відстані (Ноутбук)")
plt.xlabel("Відстань від пристрою (см)")
plt.ylabel("Середнє магнітне поле (µT)")
plt.grid(True)
plt.tight_layout()
plt.show()
