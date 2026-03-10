import random
from math import sqrt
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


class function_integral:

    def __init__(self, a, b, N):
        self.N = N
        self.a = a
        self.b = b
        self.X, self.d = np.linspace(a, b, N, retstep=True)
        self.an_integral, _ = integrate.quad(self.f, a, b, epsabs=1e-12)

    @staticmethod
    def f(x):
        return 1 / (x ** 3 + x + 10)

    @staticmethod
    def f_period(x):
        return np.sin(x)**2

    def razryvnaya(self,x):
        x_raz=self.a+(self.b-self.a)/3
        if x < x_raz:
            return x + 1  # линейная
        else:
            return 2 * x + 0.5  # другая линейная

    def rect_integral(self, h=None):
        """Метод правых прямоугольников (порядок точности 1)"""
        if h is None:
            h = self.d
        integral = np.sum(self.f(self.X[1:])) * h
        return integral

    def trap_integral(self, h=None):
        """Метод трапеций"""
        if h is None:
            h = self.d
        integral = np.sum(self.f(self.X[1:-1]))
        integral += (self.f(self.X[0]) + self.f(self.X[-1])) / 2
        integral *= h
        return integral

    def simpson_integral(self, h=None):
        """Метод Симпсона (парабол)"""
        if h is None:
            h = self.d
        x_left = self.X[:-1]
        x_right = self.X[1:]
        x_mid = (x_left + x_right) / 2
        integral = np.sum(self.f(x_left) + 4 * self.f(x_mid) + self.f(x_right))
        integral *= (h / 6)
        return integral

    def delta(self, integral_type):
        if integral_type == "rect":
            delta = abs(self.rect_integral() - self.an_integral)
        elif integral_type == "trap":
            delta = abs(self.trap_integral() - self.an_integral)
        else:
            delta = abs(self.simpson_integral() - self.an_integral)
        return delta

    def N_dependence(self, N_array, integral_type):
        delta = np.zeros(len(N_array))
        step = np.zeros(len(N_array))
        X0, d0 = self.X, self.d
        for i in range(len(N_array)):
            self.X, self.d = np.linspace(self.a, self.b, N_array[i], retstep=True)
            delta[i] = self.delta(integral_type)
            step[i] = self.d
        self.X, self.d = X0, d0
        return step, delta

    def plot_all_methods(self, N_array):
        """Построение всех методов на одном графике"""
        plt.figure(figsize=(10, 8))

        methods = [
            ("rect", "Правые прямоугольники", '-o', 'blue'),
            ("trap", "Трапеции", '-s', 'green'),
            ("simpson", "Симпсона", '-x', 'red')
        ]

        theoretical_orders = {
            "rect": 1,
            "trap": 2,
            "simpson": 4
        }

        print(f"Точное значение интеграла: {self.an_integral:.10f}")
        print("-" * 50)

        for method_type, method_name, marker, color in methods:
            step, delta = self.N_dependence(N_array, method_type)

            if len(step) > 1:
                plt.loglog(step, delta, marker, label=f'{method_name}', color=color, markersize=8, linewidth=2)

                # Оценка порядка точности
                log_h = np.log(step)
                log_delta = np.log(delta)
                p = np.polyfit(log_h, log_delta, 1)[0]

                print(f"Метод {method_name}:")
                print(f"  Фактический порядок точности: {abs(p):.2f}")
                print(f"  Теоретический порядок точности: {theoretical_orders[method_type]}")
                print(f"  Отклонение: {abs(abs(p) - theoretical_orders[method_type]):.2f}")

        plt.title("Сравнение методов численного интегрирования\n(логарифмический масштаб)", fontsize=14)
        plt.ylabel("Погрешность ", fontsize=12)
        plt.xlabel("Шаг интегрирования h", fontsize=12)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(loc='best', fontsize=9)
        plt.tight_layout()
        plt.show()
