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
    for i in range(1, len(F) - 1):
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
    if diff_type == "first":
        for i in range (len(N_array)):
            F = create_data_grid(a, b, N_array[i])

            right_diff = diff_1_right(F)
            central_diff = diff_1_central(F)

            x_right = right_diff[:, 0]
            x_central = central_diff[:, 0]

            y_analytical_right = f_diff_analytical(x_right)
            y_analytical_central = f_diff_analytical(x_central)

            delta1[i] = max(abs(y_analytical_right - right_diff[:,1]))
            delta2[i] = max(abs(y_analytical_central - central_diff[:, 1]))

        plt.loglog(N_array, delta1, '-o',label='Правая разность')
        plt.loglog(N_array, delta2, '-s',label='Центральная разность')
        plt.title("Логарифмический масштаб для погрешности 1-ой производной ")
        plt.ylabel("Погрешность")


    else:
        for i in range (len(N_array)):
            F = create_data_grid(a, b, N_array[i])

            second_diff_2 = diff_2_ord2(F)
            second_diff_4 = diff_2_ord4(F)

            x_2 = second_diff_2[:, 0]
            x_4 = second_diff_4[:, 0]

            y_analytical_2 = f_second_diff_analytical(x_2)
            y_analytical_4 = f_second_diff_analytical(x_4)

            delta1[i] = max(abs(y_analytical_2 - second_diff_2[:,1]))
            delta2[i] = max(abs(y_analytical_4 - second_diff_4[:, 1]))

        plt.loglog(N_array, delta1, '-o',label='Разность (2-ой порядок)')
        plt.loglog(N_array, delta2, '-s',label='Разность (4-ой порядок)')
        plt.title("Логарифмический масштаб для погрешности 2-ой производной ")
        plt.ylabel("Погрешность")
    plt.xlabel("N")
    plt.legend(loc='best')
    plt.show()



