import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Функція для системи диференціальних рівнянь
def energy_flow(t, y, A, B):
    return A @ y + B

# Параметри мережі
n = 5  # Кількість вузлів
A = np.array([[-0.5,  0.2,  0.1,  0.1,  0.1],
              [ 0.2, -0.6,  0.2,  0.1,  0.1],
              [ 0.1,  0.2, -0.5,  0.1,  0.1],
              [ 0.1,  0.1,  0.1, -0.4,  0.1],
              [ 0.1,  0.1,  0.1,  0.1, -0.4]])
B = np.array([0.5, -0.3, 0.1, -0.1, -0.2])

# Початкові умови
y0 = np.random.rand(n)

# Часовий інтервал
t_span = (0, 10)
t_eval = np.linspace(*t_span, 100)

# Розв’язок системи
sol = solve_ivp(energy_flow, t_span, y0, args=(A, B), t_eval=t_eval)

# Обчислення середніх значень для статистичного аналізу
mean_flows = np.mean(sol.y, axis=1)

# Візуалізація
plt.figure(figsize=(12, 6))
for i in range(n):
    plt.plot(sol.t, sol.y[i], label=f'Вузол {i+1}')
plt.xlabel('Час')
plt.ylabel('Енергія у вузлах')
plt.title('Моделювання потоків енергії в мережі')
plt.legend()
plt.grid()
plt.show()

# Порівняння зі статистичним аналізом
plt.figure(figsize=(6, 4))
plt.bar(range(1, n+1), mean_flows, color='skyblue')
plt.xlabel('Вузол')
plt.ylabel('Середнє значення енергії')
plt.title('Статистичний аналіз середніх значень енергії')
plt.grid(axis='y')
plt.show()
