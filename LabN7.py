import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def exact_solution(x, t):
    """
    Точное решение: u0(x,t) = x^2 + sinh(xt)
    """
    return x ** 2 + np.sinh(x * t)


def source_term(x, t):
    """
    Правая часть уравнения (неоднородность):
    f(x,t) = -2 + (2x^2 - t^2)*sinh(xt)
    Уравнение: 2*u_tt = u_xx + f(x,t)
    """
    return -2 + (2 * x ** 2 - t ** 2) * np.sinh(x * t)


def boundary_condition_left(t):
    """
    Граничное условие на левой границе:
    u_x(0,t) = t
    """
    return t


def boundary_condition_right(t):
    """
    Граничное условие на правой границе:
    u(1,t) + u_x(1,t) = 3 + sinh(t) + t*cosh(t)
    """
    return 3 + np.sinh(t) + t * np.cosh(t)


def initial_condition_u(x):
    """
    Начальное условие: u(x,0) = x^2
    """
    return x ** 2


def initial_condition_ut(x):
    """
    Начальное условие: u_t(x,0) = x
    """
    return x


def create_grid(Nx, Nt, T=1.0):
    """
    Создание сетки
    """
    h = 1.0 / Nx
    tau = T / Nt
    x = np.linspace(0, 1, Nx + 1)
    t = np.linspace(0, T, Nt + 1)
    return x, t, h, tau


def check_stability(h, tau):
    a = 1.0 / np.sqrt(2)
    sigma = tau / (h / a)
    is_stable = sigma <= 1.0
    return is_stable, sigma


def solve_wave_equation(Nx, Nt, T=1.0):
    x, t, h, tau = create_grid(Nx, Nt, T)
    is_stable, sigma = check_stability(h, tau)
    print(f"Параметр Куранта sigma = {sigma:.4f}")
    print(f"Условие устойчивости: {'ВЫПОЛНЯЕТСЯ' if is_stable else 'НАРУШАЕТСЯ'}")
    if not is_stable:
        print("ВНИМАНИЕ: Решение может быть неустойчивым!")
    u = np.zeros((Nt + 1, Nx + 1))
    u[0, :] = initial_condition_u(x)
    psi = initial_condition_ut(x)
    f0 = source_term(x, 0.0)
    u_xx_0 = 2.0 * np.ones_like(x)
    u[1, :] = u[0, :] + tau * psi + 0.5 * tau ** 2 * (0.5 * u_xx_0 + 0.5 * f0)
    # Временные слои с n=1 до Nt-1
    for n in range(1, Nt):
        tn = t[n]
        tn1 = t[n + 1]
        # Внутренние узлы
        for i in range(1, Nx):
            xi = x[i]
            fn = source_term(xi, tn)
            # Разностная схема
            u_xx = (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]) / h ** 2
            u[n + 1, i] = 2 * u[n, i] - u[n - 1, i] + tau ** 2 * (0.5 * u_xx + 0.5 * fn)
        g_left = boundary_condition_left(tn1)
        u[n + 1, 0] = (4 * u[n + 1, 1] - u[n + 1, 2] - 2 * h * g_left) / 3.0

        g_right = boundary_condition_right(tn1)
        u_N_minus_1 = u[n + 1, Nx - 1]
        u_N_minus_2 = u[n + 1, Nx - 2]
        u[n + 1, Nx] = (2 * h * g_right + 4 * u_N_minus_1 - u_N_minus_2) / (3 + 2 * h)
    return x, t, u


def compute_error(u_num, x, t):
    """
    Вычисление погрешности численного решения
    """
    u_exact = np.zeros_like(u_num)
    for n in range(len(t)):
        for i in range(len(x)):
            u_exact[n, i] = exact_solution(x[i], t[n])
    error = np.abs(u_num - u_exact)
    return error, u_exact


def chebyshev_norm(error):
    """
    Вычисление нормы Чебышёва
    """
    return np.max(error)


def plot_3d_solution(x, t, u, title="Численное решение"):
    """
    Построение трехмерного графика решения
    """
    X, T = np.meshgrid(x, t)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, u, cmap='viridis', alpha=0.9)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()


def plot_3d_error(x, t, error, title="Погрешность решения"):
    """
    Построение трехмерного графика погрешности
    """
    X, T = np.meshgrid(x, t)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, error, cmap='hot', alpha=0.9)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('|error|')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()


def convergence_study_x():
    """
    Исследование сходимости по x
    """
    print("\n=== Исследование сходимости по пространственной переменной x ===")
    print("tau закреплено, h меняется")
    print(f"{'Nx':<5} {'h':<10} {'tau':<10} {'Error (C-norm)':<15} {'Ratio':<10}")
    print("-" * 60)
    Nt_fixed = 200
    T = 1.0
    tau = T / Nt_fixed
    Nx_values = [20, 40, 80, 160]
    errors = []
    for Nx in Nx_values:
        h = 1.0 / Nx
        x, t, u = solve_wave_equation(Nx, Nt_fixed, T)
        error, _ = compute_error(u, x, t)
        err_norm = chebyshev_norm(error)
        errors.append(err_norm)
        if len(errors) > 1:
            ratio = errors[-2] / errors[-1]
            print(f"{Nx:<5} {h:<10.6f} {tau:<10.6f} {err_norm:<15.6e} {ratio:<10.3f}")
        else:
            print(f"{Nx:<5} {h:<10.6f} {tau:<10.6f} {err_norm:<15.6e}")

    # Построение графика сходимости
    h_values = [1.0 / Nx for Nx in Nx_values]
    plt.figure(figsize=(8, 6))
    plt.loglog(h_values, errors, 'o-', linewidth=2, markersize=8, label='Численная погрешность')
    # Эталонная линия O(h^2)
    h_ref = np.array(h_values)
    plt.loglog(h_ref, h_ref ** 2 * errors[0] / h_values[0] ** 2, '--',
               label='O(h^2)', alpha=0.5)
    plt.xlabel('Шаг по пространству h')
    plt.ylabel('Норма погрешности (C-норма)')
    plt.title('Сходимость по пространственной переменной x')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()
    return h_values, errors


def convergence_study_t():
    """
    Исследование сходимости по t
    """
    print("\n=== Исследование сходимости по временной переменной t ===")
    print("h закреплено, tau меняется")
    print(f"{'Nt':<5} {'tau':<10} {'h':<10} {'Error (C-norm)':<15} {'Ratio':<10}")
    print("-" * 60)
    Nx_fixed = 200
    T = 1.0
    h = 1.0 / Nx_fixed
    Nt_values = [20, 40, 80, 160]
    errors = []
    for Nt in Nt_values:
        tau = T / Nt
        x, t, u = solve_wave_equation(Nx_fixed, Nt, T)
        error, _ = compute_error(u, x, t)
        err_norm = chebyshev_norm(error)
        errors.append(err_norm)
        if len(errors) > 1:
            ratio = errors[-2] / errors[-1]
            print(f"{Nt:<5} {tau:<10.6f} {h:<10.6f} {err_norm:<15.6e} {ratio:<10.3f}")
        else:
            print(f"{Nt:<5} {tau:<10.6f} {h:<10.6f} {err_norm:<15.6e}")
    # Построение графика сходимости
    tau_values = [T / Nt for Nt in Nt_values]
    plt.figure(figsize=(8, 6))
    plt.loglog(tau_values, errors, 's-', linewidth=2, markersize=8, label='Численная погрешность')
    # Эталонная линия O(tau^2)
    tau_ref = np.array(tau_values)
    plt.loglog(tau_ref, tau_ref ** 2 * errors[0] / tau_values[0] ** 2, '--',
               label='O(τ^2)', alpha=0.5)
    plt.xlabel('Шаг по времени τ')
    plt.ylabel('Норма погрешности (C-норма)')
    plt.title('Сходимость по временной переменной t')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()
    return tau_values, errors


def print_formulas():
    """
    Вывод формул, использованных для численного решения
    """
    print("=" * 70)
    print("ФОРМУЛЫ ЧИСЛЕННОГО РЕШЕНИЯ")
    print("=" * 70)

    print("\n1. ИСХОДНОЕ УРАВНЕНИЕ:")
    print("   2*u_tt = u_xx - 2 + (2x^2 - t^2)*sinh(xt)")
    print("   или: u_tt = 0.5*u_xx + 0.5*f(x,t)")
    print("   где f(x,t) = -2 + (2x^2 - t^2)*sinh(xt)")

    print("\n2. РАЗНОСТНАЯ СХЕМА (второй порядок точности):")
    print("   (u_i^{n+1} - 2u_i^n + u_i^{n-1})/τ^2 =")
    print("   = 0.5*(u_{i+1}^n - 2u_i^n + u_{i-1}^n)/h^2 + 0.5*f_i^n")
    print("   ")
    print("   u_i^{n+1} = 2u_i^n - u_i^{n-1} + ")
    print("             + τ^2*[0.5*(u_{i+1}^n - 2u_i^n + u_{i-1}^n)/h^2 + 0.5*f_i^n]")

    print("\n3. НАЧАЛЬНЫЕ УСЛОВИЯ (второй порядок точности):")
    print("   u_i^0 = φ(x_i) = x_i^2")
    print("   u_i^1 = u_i^0 + τ*ψ(x_i) + 0.5*τ^2*[0.5*u_xx(x_i,0) + 0.5*f(x_i,0)]")
    print("   где ψ(x) = x, u_xx(x,0) = 2")

    print("\n4. ГРАНИЧНЫЕ УСЛОВИЯ (второй порядок точности):")
    print("   Левая граница (x=0):")
    print("   u_x(0,t) = t")
    print("   (-3u_0^n + 4u_1^n - u_2^n)/(2h) = t^n")
    print("   u_0^n = (4u_1^n - u_2^n - 2h*t^n)/3")
    print("   ")
    print("   Правая граница (x=1):")
    print("   u(1,t) + u_x(1,t) = 3 + sinh(t) + t*cosh(t)")
    print("   Аппроксимация 2-го порядка для u_x:")
    print("   u_x(1,t) ≈ (3u_N - 4u_{N-1} + u_{N-2}) / (2h)")
    print("   Подставляем в ГУ:")
    print("   u_N + (3u_N - 4u_{N-1} + u_{N-2})/(2h) = g_right")
    print("   Выражаем u_N:")
    print("   u_N = (2h·g_right + 4u_{N-1} - u_{N-2}) / (3 + 2h)")

    print("\n5. УСЛОВИЕ УСТОЙЧИВОСТИ (Куранта-Фридрихса-Леви):")
    print("   Для уравнения u_tt = a^2*u_xx + f")
    print("   a^2 = 0.5, a = 1/√2")
    print("   τ ≤ h/a = h*√2")
    print("   σ = τ/(h*√2) ≤ 1")

    print("\n6. ТОЧНОЕ РЕШЕНИЕ:")
    print("   u_0(x,t) = x^2 + sinh(xt)")

    print("=" * 70)


def main():
    """
    Основная функция, демонстрирующая все этапы решения
    """
    print("РЕШЕНИЕ СМЕШАННОЙ КРАЕВОЙ ЗАДАЧИ ДЛЯ ВОЛНОВОГО УРАВНЕНИЯ")
    print("=" * 70)

    # Вывод формул
    print_formulas()

    # Параметры расчета
    Nx = 50
    Nt = 100
    T = 1.0

    print(f"\nПАРАМЕТРЫ РАСЧЕТА:")
    print(f"  Nx = {Nx}, Nt = {Nt}")
    print(f"  h = {1.0 / Nx:.6f}, τ = {T / Nt:.6f}")

    # Решение уравнения
    print("\nВЫПОЛНЕНИЕ РАСЧЕТА...")
    x, t, u = solve_wave_equation(Nx, Nt, T)

    # Вычисление погрешности
    error, u_exact = compute_error(u, x, t)
    max_error = chebyshev_norm(error)

    print(f"\nРЕЗУЛЬТАТЫ:")
    print(f"  Максимальная погрешность (C-норма): {max_error:.6e}")

    # Построение графиков
    print("\nПОСТРОЕНИЕ ГРАФИКОВ...")

    # График численного решения
    plot_3d_solution(x, t, u, "Численное решение u(x,t)")

    # График точного решения
    plot_3d_solution(x, t, u_exact, "Точное решение u_0(x,t)")

    # График погрешности
    plot_3d_error(x, t, error, "Абсолютная погрешность |u - u_0|")

    # Исследование сходимости
    print("\nИССЛЕДОВАНИЕ СХОДИМОСТИ:")

    # Сходимость по x
    h_vals, err_x = convergence_study_x()

    # Сходимость по t
    tau_vals, err_t = convergence_study_t()

    print("\n" + "=" * 70)
    print("РАСЧЕТ ЗАВЕРШЕН УСПЕШНО!")
    print("=" * 70)


if __name__ == "__main__":
    main()