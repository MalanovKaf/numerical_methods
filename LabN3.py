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
        self.an_integral_period, _ = integrate.quad(self.f_period, a, b, epsabs=1e-12)
        self.an_integral_razryv, _ = integrate.quad(self.razryvnaya, a, b, epsabs=1e-12)

    @staticmethod
    def f(x):
        return 1 / (x ** 3 + x + 10)

    @staticmethod
    def f_period(x):
        return np.sin(x)**2

    def razryvnaya(self, x):
        x_raz = self.a + (self.b - self.a) / 3
        # Проверяем, является ли x массивом
        if isinstance(x, (list, np.ndarray)):
            # Для массивов используем векторизованное вычисление
            result = np.zeros_like(x, dtype=float)
            mask_left = x < x_raz
            mask_right = x >= x_raz
            result[mask_left] = x[mask_left] + 1
            result[mask_right] = 2 * x[mask_right] + 0.5
            return result
        else:
            # Для скалярных значений
            if x < x_raz:
                return x + 1
            else:
                return 2 * x + 0.5

    def rect_integral(self, h=None):
        """Метод правых прямоугольников (порядок точности 1)"""
        if h is None:
            h = self.d
        integral = np.sum(self.f(self.X[1:])) * h
        return integral

    def trap_integral(self, func, X=None, h=None):
        if X is None:
            X = self.X
        if h is None:
            h = self.d
        integral = np.sum(func(X[1:-1]))  # Сумма внутренних точек
        integral += (func(X[0]) + func(X[-1])) / 2  # Добавляем полусумму крайних
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
            delta = abs(self.trap_integral(self.f) - self.an_integral)
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

    def compare_functions_error(self, N_array):
        """
        Сравнение погрешности метода трапеций для трех функций
        """
        plt.figure(figsize=(12, 8))
        X0, d0 = self.X, self.d
        functions = [
            (self.f, "f(x) = 1/(x³+x+10)", 'blue', self.an_integral),
            (self.f_period, "f(x) = sin²(x)", 'green', self.an_integral_period),
            (self.razryvnaya, "Разрывная функция", 'red', self.an_integral_razryv)
        ]
        print("Анализ погрешности метода трапеций:")
        print("-" * 60)
        for func, label, color, exact_value in functions:
            errors = []
            steps = []
            for N in N_array:
                self.X, self.d = np.linspace(self.a, self.b, N, retstep=True)
                numerical = self.trap_integral(func)
                error = abs(numerical - exact_value)
                errors.append(error)
                steps.append(self.d)
            # Построение графика в логарифмическом масштабе
            plt.loglog(steps, errors, 'o-', color=color, label=label,markersize=6, linewidth=2, alpha=0.8)
            # Оценка порядка точности
            if len(steps) > 1:
                log_h = np.log(steps)
                log_error = np.log(errors)
                p, _ = np.polyfit(log_h, log_error, 1)
                print(f"\n{label}:")
                print(f"  Порядок точности: {abs(p):.3f}")
        self.X, self.d = X0, d0
        # Оформление графика
        plt.title("Зависимость погрешности метода трапеций от шага сетки\n(логарифмический масштаб)", fontsize=14, fontweight='bold')
        plt.xlabel("Шаг интегрирования h", fontsize=12)
        plt.ylabel("Погрешность ", fontsize=12)
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(loc='best', fontsize=10, framealpha=0.9)
        plt.tight_layout()
        plt.show()

