import numpy as np
import matplotlib.pyplot as plt

# Новые параметры
a = 30.0        # см
L = 5.47        # см (новое значение)
D = 1.46        # см (новое значение)
S_sum = 1e3 + 1e5  # 101000 н/с

# Массив координат x от 0 до 80 см (расширили диапазон из-за большей L)
x = np.linspace(0, 80, 300)

# Расстояние от источника до точки P
r = np.sqrt(a**2 + x**2)

# Расчет суммарного потока с новыми параметрами
flux = (S_sum / (4 * np.pi * D * r)) * np.exp(-r / L)

# Построение графика
plt.figure(figsize=(12, 7))
plt.plot(x, flux, linewidth=2.5, color='darkblue')
plt.title('Суммарная плотность потока нейтронов в точке P\n(D=1.46 см, L=5.47 см)', fontsize=14)
plt.xlabel('Расстояние x (см)', fontsize=12)
plt.ylabel('Плотность потока (нейтрон/см²·с)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.7)



# Отметка значения при x=30

plt.tight_layout()
plt.show()

