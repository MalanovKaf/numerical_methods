import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """Исходная функция: arcsin(th(x)) - x/2 + 3 = 0"""
    return math.asin(math.tanh(x)) - x/2 + 3

def f_derivative(x):
    """Производная функции f(x)"""
    return 1/math.cosh(x) - 1/2

def f_vectorized(x):
    """Векторизованная версия функции для numpy массивов"""
    return np.arcsin(np.tanh(x)) - x/2 + 3

def bisection(f, a, b, delta, max_iters=100):
    """ Метод дихотомии (бисекции) для поиска корня уравнения f(x)=0 """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(f"На интервале [{a}, {b}] нет корня или четное количество корней")
    xs = []  # история приближений
    iteration = 0
    while (b - a) / 2 > delta and iteration < max_iters:
        c = (a + b) / 2
        xs.append(c)
        fc = f(c)
        if fc == 0:
            return c, xs
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        iteration += 1
    if iteration >= max_iters:
        raise Exception(f"Превышено максимальное количество итераций ({max_iters}) в методе дихотомии")
    return xs[-1], xs

def find_initial_interval(step=0.1):
    """ Интервал (a, b) в (0, 10), на котором f(a)*f(b) < 0 """
    x = 0
    while x <= 10:
        fx = f(x)
        if fx == 0:
            return (x, x)  # интервал нулевой длины
        x_next = x + step
        if x_next > 10:
            break
        fx_next = f(x_next)
        if fx_next == 0:
            return (x_next, x_next)  # интервал нулевой длины
        if fx * fx_next < 0:
            return (x, x_next)
        x = x_next
    raise ValueError("Не удалось найти интервал")


def newton(f, f_der, x0, delta, max_iters=1000):
    """ Метод Ньютона (касательных) для поиска корня уравнения f(x)=0 """
    xs = []
    x_current = x0
    iteration = 0
    while iteration < max_iters:
        fx = f(x_current)
        fdx = f_der(x_current)
        if fdx == 0:
            raise Exception(f"Производная равна нулю в точке x = {x_current}")
        x_next = x_current - fx / fdx
        xs.append(x_next)
        if abs(x_next - x_current) < delta*abs(x_next):
            break
        x_current = x_next
        iteration += 1

    if iteration >= max_iters:
        raise Exception(f"Превышено максимальное количество итераций ({max_iters}) в методе Ньютона. "
                        f"Последнее значение f(x) = {f(x_current):.2e}")
    return xs[-1], xs

def plot_function():
    """Построение графика функции f(x)"""
    x = np.linspace(-20, 20, 1000)
    y = f_vectorized(x)

    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'b-', linewidth=2, label='f(x) = arcsin(th(x)) - x/2 + 3')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('График функции f(x)', fontsize=14)
    plt.legend()
    try:
        root, _ = bisection(f, 0, 10, 1e-6)
        plt.plot(root, 0, 'ro', markersize=10, label=f'Корень: x ≈ {root:.6f}')
        plt.legend()
    except:
        pass
    plt.tight_layout()
    plt.show()

def study_bisection_method():
    """Исследование метода дихотомии на разных интервалах"""
    print("ИССЛЕДОВАНИЕ МЕТОДА ДИХОТОМИИ")
    delta = 1e-6  # фиксированная абсолютная погрешность
    intervals = [
        (0, 10),
        (8, 10),
        (9, 10),
        (9, 9.5),
        (0, 8)
    ]
    results = []
    for a, b in intervals:
        print(f"\nИнтервал [{a}, {b}]:")
        try:
            root, history = bisection(f, a, b, delta)
            print(f"  Корень найден: x = {root:.8f}")
            print(f"  Количество итераций: {len(history)}")
            print(f"  Значение функции f(x) = {f(root):.2e}")
            results.append((a, b, root, len(history), True))
        except Exception as e:
            print(f"  Ошибка: {e}")
            results.append((a, b, None, 0, False))
    return results

def study_newton_method():
    """Исследование метода Ньютона с разными начальными приближениями"""
    print("ИССЛЕДОВАНИЕ МЕТОДА НЬЮТОНА")
    delta = 1e-6  # фиксированная абсолютная погрешность
    initial_guesses = [
        0.0,
        1.0,
        5.0,
        9.0,
        10.0
    ]
    results = []
    for x0 in initial_guesses:
        print(f"\nНачальное приближение x0 = {x0}:")
        try:
            root, history = newton(f, f_derivative, x0, delta)
            print(f"  Корень найден: x = {root:.8f}")
            print(f"  Количество итераций: {len(history)}")
            print(f"  Значение функции f(x) = {f(root):.2e}")
            results.append((x0, root, len(history) , history, True))
        except Exception as e:
            print(f"  Ошибка: {e}")
            results.append((x0, None, 0, None, False))
    return results

plot_function()
study_bisection_method()
study_newton_method()
