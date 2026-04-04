import matplotlib.pyplot as plt
import numpy as np

# Данные из таблицы
# Энергия в кэВ (заменил текстовые значения на числа)
energy_kev = [
    0.025,   # Thermal (~0.025 eV -> 0.000025 keV, но для логарифмической шкалы удобнее 0.025)
    2,
    25,
    144,
    250,
    565,
    1200,
    2500,
    2800,
    3200,
    5000,
    14800,
    19000
]

# Значения h*phi(10) в пЗв·см²
h_phi = [
    10.6,
    7.7,
    19.3,
    127,
    203,
    343,
    425,
    416,
    413,
    411,
    405,
    536,
    584
]

# Создание графика
plt.figure(figsize=(10, 6))

# Построение с маркерами
plt.semilogx(energy_kev, h_phi, 'o-', color='b', linewidth=2, markersize=8, label='h*ф(10; E)')

# Настройка внешнего вида
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xlabel('Энергия нейтронов E, кэВ', fontsize=12)
plt.ylabel('h*ф(10; E), пЗв·см²', fontsize=12)
plt.title('Зависимость коэффициента перехода h*ф от энергии нейтронов', fontsize=14)

# Подписи осей
plt.xlim(0.01, 20000)  # От 0.01 кэВ до 20 МэВ
plt.ylim(0, 450)

# Добавление подписей точек
for i, (x, y) in enumerate(zip(energy_kev, h_phi)):
    if i == 0 or i == 1 or i == 2 or i == 3:
        plt.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)
    elif i == 4:
        plt.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(5,-10), ha='left', fontsize=8)

# Легенда
plt.legend()

# Показать график
plt.tight_layout()
plt.show()

# Вывод максимального значения
max_idx = np.argmax(h_phi)
print(f"Максимальное значение h*φ(10) = {h_phi[max_idx]} пЗв·см² при энергии E = {energy_kev[max_idx]} кэВ")