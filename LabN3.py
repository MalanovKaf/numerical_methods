import random
from math import sqrt
import sys
import numpy as np
import matplotlib.pyplot as plt

class function_integral:

    def __init__(self,a,b,N):
        self.N=N
        self.a=a
        self.b=b
        self.X, self.d = np.linspace(a,b,N,retstep=True)
        self.an_integral=0.2

    @staticmethod
    def f(x):
        return 1/(x**3+x+10)

    def rect_integral(self, h=None):
        """Метод прямоугольников (средние точки)"""
        if h is None:
            h = self.d
        x_midpoints = (self.X[:-1] + self.X[1:]) / 2
        integral = np.sum(self.f(x_midpoints)) * h
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

        if self.N % 2 != 0:
            print("Предупреждение: Метод Симпсона работает лучше при четном N")

        x_left = self.X[:-1]
        x_right = self.X[1:]
        x_mid = (x_left + x_right) / 2

        integral = np.sum(self.f(x_left) + 4 * self.f(x_mid) + self.f(x_right))
        integral *= (h / 6)
        return integral

    def delta(self,integral_type):
        if integral_type=="rect":
            delta=abs(self.rect_integral()-self.an_integral)
        elif integral_type=="trap":
            delta = abs(self.trap_integral() - self.an_integral)
        else:
            delta = abs(self.simpson_integral() - self.an_integral)
        return delta

    def N_dependece(self,N_array,integral_type):
        delta = np.zeros(len(N_array))
        step=np.zeros(len(N_array))
        X0,d0 = self.X, self.d
        for i in range (len(N_array)):
            self.X, self.d = np.linspace(self.a, self.b, N_array[i], retstep=True)
            delta[i]=self.delta(integral_type)
            step[i]=self.d
        self.X, self.d = X0, d0
        return step, delta

    def plot(self,h_array,delta,integral_type):
        if integral_type=="rect":
            plt.loglog(h_array, delta, '-o',label='Интеграл по формуле прямоугольников')
            log_h = np.log(h_array)
            log_delta = np.log(delta)
            p = np.polyfit(log_h, log_delta, 1)[0]
            print(f"Порядок точности для формулы прямоугольника: {abs(p):.2f}")

        elif integral_type=="trap":
            plt.loglog(h_array, delta, '-s', label='Интеграл по формуле трапеции')
            log_h = np.log(h_array)
            log_delta = np.log(delta)
            p = np.polyfit(log_h, log_delta, 1)[0]
            print(f"Порядок точности для формулы трапеции: {abs(p):.2f}")

        else:
            plt.loglog(h_array, delta, '-x', label='Интеграл по формуле Симпсона')
            log_h = np.log(h_array)
            log_delta = np.log(delta)
            p = np.polyfit(log_h, log_delta, 1)[0]
            print(f"Порядок точности для формулы Симпсона: {abs(p):.2f}")

        plt.title("Логарифмический масштаб погрешности")
        plt.ylabel("Погрешность")
        plt.xlabel("h")
        plt.legend(loc='best')
        plt.show()


















