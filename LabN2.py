import matplotlib.pyplot as plt
import numpy as np
import math


def nodes(a,b,N):
    return np.linspace(a, b, N)


def f_diff_analytical(x):
    """
    Аналитическая производная функции f(x)
    f(x) = (arctan(ln(x^2+1)+1))^2
    """
    log_part = np.log(x ** 2 + 1) + 1
    arctan_part = np.arctan(log_part)
    diff = 2 * arctan_part * (1 / (1 + log_part ** 2)) * (1 / (x ** 2 + 1)) * (2 * x)
    return diff


def f_second_diff_analytical(x):
    """
        Аналитическая вторая производная функции f(x)
        Вычисляется численно через градиент от первой производной для простоты
    """
    h = 1e-5
    return (f_diff_analytical(x + h) - f_diff_analytical(x - h)) / (2 * h)


def f(x):
    """Исходная функция"""
    return (np.arctan(np.log(x**2+1)+1))**2


def create_data_grid(a, b, N):
    """Создает сеточные данные: координаты и значения функции"""
    x = nodes(a, b, N)
    y = f(x)
    return np.column_stack((x, y))


def diff_1_right(F):
    h=F[1][0]-F[0][0]
    diff=[]
    for i in range (len(F)-1):
        derivative = (F[i + 1][1] - F[i][1]) / h
        diff.append([F[i][0], derivative])
    return np.array(diff)


def diff_1_central(F):
    h=F[1][0]-F[0][0]
    diff = []
    for i in range(len(F)):
        if i==0:
            derivative=(2*F[i+1][1]-1.5*F[i][1]-0.5*F[i+2][1])/h
        elif i==len(F)-1:
            derivative=(1.5*F[i][1]-2*F[i-1][1]+0.5*F[i-2][1])/h
        else:
            derivative = (F[i + 1][1] - F[i - 1][1]) / (2 * h)
        diff.append([F[i][0], derivative])
    return np.array(diff)


def diff_2_ord2(F):
    h = F[1][0] - F[0][0]
    diff = []
    for i in range(1, len(F) - 1):
        derivative = (F[i + 1][1] - 2 * F[i][1] + F[i - 1][1]) / (h ** 2)
        diff.append([F[i][0], derivative])
    return np.array(diff)


def diff_2_ord4(F):
    h = F[1][0] - F[0][0]
    diff = []
    for i in range(2, len(F) - 2):
        derivative = (4*(F[i + 1][1] - 2 * F[i][1] + F[i - 1][1]) / (3*(h ** 2)))-((F[i + 2][1] - 2 * F[i][1] + F[i - 2][1]) / (12*(h ** 2)))
        diff.append([F[i][0], derivative])
    return np.array(diff)


def compare_diff_plot(F, diff_type,N):
    x_n = np.linspace(F[0, 0], F[-1, 0], N)
    if diff_type == "first":
        right_diff = diff_1_right(F)
        central_diff = diff_1_central(F)
        y_analytical = f_diff_analytical(x_n)
        plt.plot(x_n, y_analytical, '-o',label='Аналитическая производная')
        plt.plot(right_diff[:, 0], right_diff[:, 1], '-s',label='Правая разность')
        plt.plot(central_diff[:, 0], central_diff[:, 1], '-x', label='Центральная разность')
        plt.title("Сравнение методов вычисления первой производной")
        plt.ylabel("f'(x)")
    else:
        second_diff_2 = diff_2_ord2(F)
        second_diff_4 = diff_2_ord4(F)
        y_analytical = f_second_diff_analytical(x_n)
        plt.plot(x_n, y_analytical, '-o',label='Аналитическая вторая производная')
        plt.plot(second_diff_2[:, 0], second_diff_2[:, 1], '-s',label='Центральная разность(2-ой порядок)')
        plt.plot(second_diff_4[:, 0], second_diff_4[:, 1], '-x', label='Центральная разность(4-ой порядок)')
        plt.title("Сравнение методов вычисления второй производной")
        plt.ylabel("f''(x)")
    plt.xlabel("x")
    plt.legend(loc='best')
    plt.show()


def delta(a,b,N_array,diff_type):

    delta1 = np.zeros(len(N_array))
    delta2 = np.zeros(len(N_array))

    h_array=np.zeros(len(N_array))

    if diff_type == "first":
        for i in range (len(N_array)):
            F = create_data_grid(a, b, N_array[i])
            h=np.linspace(a,b,N_array[i],retstep=True)[1]

            right_diff = diff_1_right(F)
            central_diff = diff_1_central(F)

            x_right = right_diff[:, 0]
            x_central = central_diff[:, 0]

            y_analytical_right = f_diff_analytical(x_right)
            y_analytical_central = f_diff_analytical(x_central)

            delta1[i] = max(abs(y_analytical_right - right_diff[:,1]))
            delta2[i] = max(abs(y_analytical_central - central_diff[:, 1]))
            h_array[i]=h

        # Определение порядков точности через линейный график как коэф. угла наклона
        log_h = np.log(h_array)
        log_delta1 = np.log(delta1)
        log_delta2 = np.log(delta2)

        # Аппроксимация прямой: log(δ) = p*log(h) + const
        p1 = np.polyfit(log_h, log_delta1, 1)[0]
        p2 = np.polyfit(log_h, log_delta2, 1)[0]
        print(f"Порядок точности для правой разности: {abs(p1):.2f}")
        print(f"Порядок точности для центральной разности: {abs(p2):.2f}")

        # Оценка максимального значение третьей производной
        x_test = np.linspace(a, b, 1000)
        f_third_max = max(abs((f_second_diff_analytical(x_test + 1e-5) -f_second_diff_analytical(x_test - 1e-5)) / (2e-5)))
        epsilon = 1e-15

        h_opt_theory = (epsilon / f_third_max) ** (1 / 3)
        h_opt_fact = h_array[np.argmin(delta2)]

        plt.loglog(h_array, delta1, '-o',label='Правая разность')
        plt.loglog(h_array, delta2, '-s',label='Центральная разность')

        plt.axvline(x=h_opt_theory, color='r', linestyle='--', label=f'Теоретический h_opt = {h_opt_theory:.2e}')
        plt.axvline(x=h_opt_fact, color='g', linestyle=':', label=f'Фактический h_opt = {h_opt_fact:.2e}')

        plt.title("Логарифмический масштаб для погрешности 1-ой производной ")
        plt.ylabel("Погрешность")


    else:
        for i in range (len(N_array)):
            F = create_data_grid(a, b, N_array[i])
            h = np.linspace(a, b, N_array[i], retstep=True)[1]

            second_diff_2 = diff_2_ord2(F)
            second_diff_4 = diff_2_ord4(F)

            x_2 = second_diff_2[:, 0]
            x_4 = second_diff_4[:, 0]

            y_analytical_2 = f_second_diff_analytical(x_2)
            y_analytical_4 = f_second_diff_analytical(x_4)

            delta1[i] = max(abs(y_analytical_2 - second_diff_2[:,1]))
            delta2[i] = max(abs(y_analytical_4 - second_diff_4[:, 1]))
            h_array[i] = h

        # Определение порядков точности
        log_h = np.log(h_array)
        log_delta1 = np.log(delta1)
        log_delta2 = np.log(delta2)

        p1 = np.polyfit(log_h, log_delta1, 1)[0]
        p2 = np.polyfit(log_h, log_delta2, 1)[0]

        print(f"Порядок точности для 2-го порядка: {abs(p1):.2f}")
        print(f"Порядок точности для 4-го порядка: {abs(p2):.2f}")

        f_4th_max = 1.0  # тут требуется более точная оценка

        epsilon = 1e-15
        h_opt_2nd = (epsilon / f_4th_max) ** (1 / 4)
        h_opt_4th = (epsilon / f_4th_max) ** (1 / 6)

        # Фактические оптимальные шаги
        h_opt_fact_2nd = h_array[np.argmin(delta1)]
        h_opt_fact_4th = h_array[np.argmin(delta2)]

        plt.loglog(h_array, delta1, '-o',label='Разность (2-ой порядок)')
        plt.loglog(h_array, delta2, '-s',label='Разность (4-ой порядок)')
        plt.axvline(x=h_opt_fact_2nd, color='r', linestyle='--', label=f'Факт. h_opt (2-й пор.) = {h_opt_fact_2nd:.2e}')
        plt.axvline(x=h_opt_2nd, color='g', linestyle=':', label=f'Теор. h_opt (2-й пор.) = {h_opt_2nd:.2e}')
        plt.title("Логарифмический масштаб для погрешности 2-ой производной ")
        plt.ylabel("Погрешность")

    plt.xlabel("h")
    plt.legend(loc='best')
    plt.show()



