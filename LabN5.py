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
    for i in range(1, min(4, n_steps)):
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
    """
    Построение графиков численных решений ОДУ разными методами

    Параметры:
    h - шаг сетки
    x_end - конечная точка интегрирования
    """
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
    h_values = np.array([0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005,10e-6])
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
