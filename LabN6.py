import numpy as np
import matplotlib.pyplot as plt


def exact_solution(x):
    return np.cos(x) + 2 * np.log(1 + x)


def coeff_u(x):
    p = 1.0 / (1 + x)
    q = np.tan(x) / (1 + x)
    f = (2 * np.tan(x) / (1 + x)) * np.log(1 + x) - np.cos(x)
    return p, q, f


def progonka(a, b, c, d):
    n = len(b)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] + a[i] * alpha[i - 1]
        alpha[i] = -c[i] / denom
        beta[i] = (d[i] - a[i] * beta[i - 1]) / denom
    y = np.zeros(n)
    y[n - 1] = (d[n - 1] - a[n - 1] * beta[n - 2]) / (b[n - 1] + a[n - 1] * alpha[n - 2])
    for i in range(n - 2, -1, -1):
        y[i] = alpha[i] * y[i + 1] + beta[i]
    return y


def solve_bvp(N, order_bc=1):
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    n = N + 1
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    p0, q0, f0 = coeff_u(0.0)

    if order_bc == 1:
        # Аппроксимация 1-го порядка: u'(0) ≈ (u1 - u0)/h
        # u0 - (u1 - u0)/h = -1  =>  u0*(1 + 1/h) - u1*(1/h) = -1
        a[0] = 0.0
        b[0] = 1.0 + 1.0 / h
        c[0] = -1.0 / h
        d[0] = -1.0
    elif order_bc == 2:
        # Аппроксимация 2-го порядка (метод фиктивного узла)
        # u_{-1} = u1 - 2h*(u0 + 1)
        a[0] = 0.0
        b[0] = -2.0 / h ** 2 - 2.0 / h + p0 + q0
        c[0] = 2.0 / h ** 2
        d[0] = f0 + 2.0 / h - p0
    else:
        raise ValueError("order_bc должен быть 1 или 2")

    for i in range(1, N):
        xi = x[i]
        p, q, f = coeff_u(xi)
        a[i] = 1.0 / h ** 2 - p / (2 * h)
        b[i] = -2.0 / h ** 2 + q
        c[i] = 1.0 / h ** 2 + p / (2 * h)
        d[i] = f

    a[N] = 0.0
    b[N] = 1.0
    c[N] = 0.0
    d[N] = exact_solution(1.0)

    u = progonka(a, b, c, d)
    return x, u


def compute_error(u_num, x):
    u_ex = exact_solution(x)
    return np.max(np.abs(u_num - u_ex))  # C-норма (максимум)

def main():
    print("Исследование сходимости...")
    print(f"{'N':<5} {'h':<10} {'Err (1st)':<15} {'Err (2nd)':<15} {'Ratio1':<8} {'Ratio2':<8}")
    print("-" * 75)

    N_values = [10, 20, 40, 80, 160]
    errors_1st, errors_2nd, h_values = [], [], []
    prev_e1, prev_e2 = None, None

    for N in N_values:
        h = 1.0 / N
        h_values.append(h)

        # Решаем для обоих порядков
        _, u1 = solve_bvp(N, order_bc=1)
        err1 = compute_error(u1, np.linspace(0, 1, N + 1))
        errors_1st.append(err1)

        _, u2 = solve_bvp(N, order_bc=2)
        err2 = compute_error(u2, np.linspace(0, 1, N + 1))
        errors_2nd.append(err2)

        r1 = prev_e1 / err1 if prev_e1 else None
        r2 = prev_e2 / err2 if prev_e2 else None

        print(f"{N:<5} {h:<10.5f} {err1:<15.5e} {err2:<15.5e} "
              f"{r1 if r1 else '-':<8} {r2 if r2 else '-':<8}")

        prev_e1, prev_e2 = err1, err2

    plt.figure(figsize=(9, 6))
    h_arr = np.array(h_values)

    plt.loglog(h_arr, errors_1st, 'bo-', lw=2, ms=6, label='ГУ 1-го порядка')
    plt.loglog(h_arr, errors_2nd, 'rs-', lw=2, ms=6, label='ГУ 2-го порядка')

    # Эталонные линии
    plt.loglog(h_arr, h_arr * errors_1st[0] / h_values[0], 'k--', alpha=0.4, label='O(h)')
    plt.loglog(h_arr, h_arr ** 2 * errors_2nd[0] / h_values[0] ** 2, 'g--', alpha=0.4, label='O(h²)')

    plt.xlabel('Шаг сетки $h$')
    plt.ylabel('Максимальная погрешность $||u - u_{exact}||_C$')
    plt.title('Сходимость разностной схемы')
    plt.legend()
    plt.grid(True, which='both', ls=':')
    plt.show()

    print("\nОценка по правилу Рунге (N=40, 2-й порядок ГУ):")
    x_c, u_c = solve_bvp(40, order_bc=2)
    x_f, u_f = solve_bvp(80, order_bc=2)


    u_f_on_c = u_f[::2]
    runge_est = np.abs(u_c - u_f_on_c) / 3.0
    true_err = compute_error(u_c, x_c)
    true_err_fine = compute_error(u_f, x_f)
    print(f"Оценка Рунге (max): {np.max(runge_est):.5e}")
    print(f"Точная погрешность: {true_err_fine:.5e}")
    print(f"Отношение: {true_err_fine / np.max(runge_est):.3f}")


if __name__ == "__main__":
    main()