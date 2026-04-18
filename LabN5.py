import numpy as np
import matplotlib.pyplot as plt


def system(x, y):
    y1, y2 = y
    dy1 = y2
    dy2 = (-1/(1+x)) * y2 - (np.tan(x)/(1+x)) * y1 + (2 * np.tan(x)/(1+x)) * np.log(1+x) - np.cos(x)
    return np.array([dy1, dy2])


def exact_solution(x):
    return np.cos(x) + 2 * np.log(1 + x)


def euler_method(f, x0, y0, h, x_end):
    n_steps = int((x_end - x0) / h) + 1
    x_vals = np.linspace(x0, x_end, n_steps)
    y_vals = np.zeros((n_steps, len(y0)))
    y_vals[0] = y0
    for i in range(1, n_steps):
        y_vals[i] = y_vals[i - 1] + h * f(x_vals[i - 1], y_vals[i - 1])
    return x_vals, y_vals


def runge_kutta_4(f, x0, y0, h, x_end):
    n_steps = int((x_end - x0) / h) + 1
    x_vals = np.linspace(x0, x_end, n_steps)
    y_vals = np.zeros((n_steps, len(y0)))
    y_vals[0] = y0
    for i in range(1, n_steps):
        x = x_vals[i - 1]
        y = y_vals[i - 1]
        k1 = f(x, y)
        k2 = f(x + h / 2, y + (h / 2) * k1)
        k3 = f(x + h / 2, y + (h / 2) * k2)
        k4 = f(x + h, y + h * k3)
        y_vals[i] = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_vals, y_vals


def adams_3rd_order(f, x0, y0, h, x_end):
    n_steps = int((x_end - x0) / h) + 1
    x_vals = np.linspace(x0, x_end, n_steps)
    y_vals = np.zeros((n_steps, len(y0)))
    y_vals[0] = y0
    for i in range(1, 4):
        if i >= n_steps:
            break
        x = x_vals[i - 1]
        y = y_vals[i - 1]
        k1 = f(x, y)
        k2 = f(x + h / 2, y + (h / 2) * k1)
        k3 = f(x + h / 2, y + (h / 2) * k2)
        k4 = f(x + h, y + h * k3)
        y_vals[i] = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    for i in range(3, n_steps - 1):
        f_n = f(x_vals[i], y_vals[i])
        f_n1 = f(x_vals[i - 1], y_vals[i - 1])
        f_n2 = f(x_vals[i - 2], y_vals[i - 2])

        y_vals[i + 1] = y_vals[i] + (h / 12) * (23 * f_n - 16 * f_n1 + 5 * f_n2)
    return x_vals, y_vals


def plot_numerical_solutions(h=0.05, x_end=1):
    x0 = 0
    y0 = np.array([1.0, 2.0])
    # Решение разными методами
    x_euler, y_euler = euler_method(system, x0, y0, h, x_end)
    x_rk4, y_rk4 = runge_kutta_4(system, x0, y0, h, x_end)
    x_adams, y_adams = adams_3rd_order(system, x0, y0, h, x_end)
    # Точное решение
    x_exact = np.linspace(x0, x_end, 500)
    y_exact = exact_solution(x_exact)
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(x_exact, y_exact, 'k-', label='Точное решение', linewidth=2)
    plt.plot(x_euler, y_euler[:, 0], 'ro-', label='Метод Эйлера', markersize=3, linewidth=1)
    plt.plot(x_rk4, y_rk4[:, 0], 'gs-', label='Рунге-Кутта 4 порядка', markersize=3, linewidth=1)
    plt.plot(x_adams, y_adams[:, 0], 'b^-', label='Адамс 3 порядка', markersize=3, linewidth=1)
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title(f'Численные решения ОДУ (шаг h = {h})')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_error_vs_step():
    """Построение графика зависимости погрешности от шага"""
    x0, x_end = 0, 1
    y0 = np.array([1.0, 2.0])
    h_values = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
    errors_euler, errors_rk4, errors_adams = [], [], []
    exact_val = exact_solution(x_end)
    for h in h_values:
        _, y_euler = euler_method(system, x0, y0, h, x_end)
        _, y_rk4 = runge_kutta_4(system, x0, y0, h, x_end)
        _, y_adams = adams_3rd_order(system, x0, y0, h, x_end)
        errors_euler.append(abs(y_euler[-1, 0] - exact_val))
        errors_rk4.append(abs(y_rk4[-1, 0] - exact_val))
        errors_adams.append(abs(y_adams[-1, 0] - exact_val))
    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, errors_euler, 'ro-', label='Эйлер', linewidth=2, markersize=6)
    plt.loglog(h_values, errors_rk4, 'gs-', label='Рунге-Кутта 4', linewidth=2, markersize=6)
    plt.loglog(h_values, errors_adams, 'b^-', label='Адамс 3', linewidth=2, markersize=6)
    plt.xlabel('Шаг h', fontsize=12)
    plt.ylabel('Погрешность', fontsize=12)
    plt.title('Зависимость погрешности от шага интегрирования', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def runge_error_estimate(y_h, y_h2, p):
    """Оценка погрешности по правилу Рунге"""
    return (y_h2 - y_h) / (2**p - 1)


def runge_error_analysis(h=0.05, x_end=1):
    x0 = 0
    y0 = np.array([1.0, 2.0])
    exact_val = exact_solution(x_end)
    h2 = h / 2
    # 1. Метод Эйлера (порядок точности p = 1)
    print("\n МЕТОД ЭЙЛЕРА (p = 1)")
    print("-" * 40)
    _, y_h = euler_method(system, x0, y0, h, x_end)
    y_h_val = y_h[-1, 0]
    true_error_h = abs(y_h_val - exact_val)
    _, y_h2 = euler_method(system, x0, y0, h2, x_end)
    y_h2_val = y_h2[-1, 0]
    true_error_h2 = abs(y_h2_val - exact_val)
    # Оценка погрешности по Рунге для метода 1-го порядка
    runge_error = runge_error_estimate(y_h_val, y_h2_val, p=1)
    runge_error_abs = abs(runge_error)
    print(f"Решение с шагом h      = {h}:     y ≈ {y_h_val:.10f}")
    print(f"Решение с шагом h/2    = {h2}:   y ≈ {y_h2_val:.10f}")
    print(f"Точная погрешность (h/2)       : {true_error_h2:.10e}")
    print(f"Оценка Рунге (h/2)             : {runge_error_abs:.10e}")
    print(f"Отношение оценка/точная      : {runge_error_abs / true_error_h2:.4f}")


    # 2. Метод Рунге-Кутты 4 порядка (p = 4)
    print("\n МЕТОД РУНГЕ-КУТТЫ 4-ГО ПОРЯДКА (p = 4)")
    print("-" * 40)

    _, y_h = runge_kutta_4(system, x0, y0, h, x_end)
    y_h_val = y_h[-1, 0]
    true_error_h = abs(y_h_val - exact_val)

    _, y_h2 = runge_kutta_4(system, x0, y0, h2, x_end)
    y_h2_val = y_h2[-1, 0]
    true_error_h2 = abs(y_h2_val - exact_val)
    # Оценка погрешности по Рунге для метода 4-го порядка
    runge_error = runge_error_estimate(y_h_val, y_h2_val, p=4)
    runge_error_abs = abs(runge_error)

    print(f"Решение с шагом h      = {h}:     y ≈ {y_h_val:.10f}")
    print(f"Решение с шагом h/2    = {h2}:   y ≈ {y_h2_val:.10f}")
    print(f"Точная погрешность (h/2)       : {true_error_h2:.10e}")
    print(f"Оценка Рунге (h/2)             : {runge_error_abs:.10e}")
    print(f"Отношение оценка/точная      : {runge_error_abs / true_error_h2:.4f}")


    # 3. Метод Адамса 3-го порядка (p = 3)
    print("\n МЕТОД АДАМСА 3-ГО ПОРЯДКА (p = 3)")
    print("-" * 40)

    _, y_h = adams_3rd_order(system, x0, y0, h, x_end)
    y_h_val = y_h[-1, 0]
    true_error_h = abs(y_h_val - exact_val)

    _, y_h2 = adams_3rd_order(system, x0, y0, h2, x_end)
    y_h2_val = y_h2[-1, 0]
    true_error_h2 = abs(y_h2_val - exact_val)
    # Оценка погрешности по Рунге для метода 3-го порядка
    runge_error = runge_error_estimate(y_h_val, y_h2_val, p=3)
    runge_error_abs = abs(runge_error)

    print(f"Решение с шагом h      = {h}:     y ≈ {y_h_val:.10f}")
    print(f"Решение с шагом h/2    = {h2}:   y ≈ {y_h2_val:.10f}")
    print(f"Точная погрешность (h/2)       : {true_error_h2:.10e}")
    print(f"Оценка Рунге (h/2)             : {runge_error_abs:.10e}")
    print(f"Отношение оценка/точная      : {runge_error_abs / true_error_h2:.4f}")

'''
def auto_runge_kutta_4(f, x0, y0, h, x_end,delta):
    x_vals = np.linspace(x0, x_end, n_steps)
    y_vals = np.zeros((n_steps, len(y0)))
    y_vals[0] = y0
    for i in range(1, n_steps):
        x = x_vals[i - 1]
        y = y_vals[i - 1]
        k1 = f(x, y)
        k2 = f(x + h / 2, y + (h / 2) * k1)
        k3 = f(x + h / 2, y + (h / 2) * k2)
        k4 = f(x + h, y + h * k3)
        y_vals[i] = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        if (y_vals[i]-y_true)>delta:
            h=h/2 есть в интернете изменение шага 
            график решения 
            график изменения шага при изменении (шаг от итерации) 
    return x_vals, y_vals
'''
plot_numerical_solutions()
plot_error_vs_step()
runge_error_analysis()