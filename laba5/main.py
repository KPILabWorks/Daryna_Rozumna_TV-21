import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Завантаження набору даних
url = "heart.csv"
data = pd.read_csv(url)

# Перегляд перших рядків датасету
print(data.head())

# Візуалізація кореляції між ознаками
df_corr = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Кореляція між ознаками")
plt.show()

# Розділення ознак (X) та цільової змінної (y)
X = data.drop(columns=['target'])  # Усі колонки, окрім 'target'
y = data['target']  # Цільова змінна (0 - немає хвороби, 1 - є хвороба)

# Розділення на тренувальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабування ознак (нормалізація)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Створення та навчання моделі Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Передбачення на тестовому наборі
y_pred = model.predict(X_test)

# Оцінка моделі
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Точність моделі: {accuracy:.2f}')
print('Матриця невідповідностей:')
print(conf_matrix)
print('Звіт класифікації:')
print(class_report)

# Візуалізація матриці невідповідностей
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Немає хвороби', 'Є хвороба'], 
            yticklabels=['Немає хвороби', 'Є хвороба'])
plt.xlabel('Передбачений клас')
plt.ylabel('Справжній клас')
plt.title('Матриця невідповідностей')
plt.show()