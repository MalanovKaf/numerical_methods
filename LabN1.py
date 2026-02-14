import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import lagrange


def nodes(a,b,N,m):
    if m==0:
        return np.linspace(a,b,N)
    else:
        return np.array([(a + b) / 2 + ((b - a) * np.cos(((2 * k - 1) * np.pi) / (2 * N))) / 2 for k in range(1, N + 1)] [::-1]) #Узлы Чебышева


def f(x):
    """Исходная функция"""
    return np.sin((x / 3) + np.exp((np.sin(x / 3)) ** 2))


def Lagrange(x_values,y_values,x_eval):
    X=np.array(x_values)
    Y=np.array(y_values)
    x_eval=np.array(x_eval)
    n=len(x_values)
    result = np.zeros_like(x_eval, dtype=float)
    for i in range (n):
        L_i = np.ones_like(x_eval, dtype=float) #массив для i-го базисного полинома
        for j in range (n):
            if j!=i:
                L_i*=(x_eval-x_values[j])/(x_values[i]-x_values[j])
        result+=y_values[i]*L_i
    return result #возвращает значения полинома Лагранжа в заданных точках x


def interpolation(a,b,N,m,z):
    X=nodes(a,b,N,m)
    Y=f(X)
    x_test=(X[1:]+X[:-1])/2
    if z==0:
        y_values=Lagrange(X,Y,x_test)
    else:
        poly=lagrange(X,Y)
        y_values=poly(x_test)
    plt.plot(X, Y, "-o", label="Искомая функция")
    plt.plot(x_test,y_values , "-x", label="Полинома Лагранжа")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Два совмещенных графика и легенда")
    plt.show()
    Y1=f(x_test)
    print(max(abs(Y1-y_values)))


def delta_max(a,b,N_array,m,z):
    delta = np.zeros(len(N_array))
    for i in range (len(N_array)):
        X = nodes(a, b, N_array[i], m)
        Y = f(X)
        x_test = (X[1:] + X[:-1]) / 2
        if z == 0:
            y_values = Lagrange(X, Y, x_test)
        else:
            poly = lagrange(X, Y)
            y_values = poly(x_test)
        Y1 = f(x_test)
        delta[i] = max(abs(Y1 - y_values))
    plt.loglog(N_array,delta)
    plt.xlabel("N")
    plt.ylabel("Погрешность")
    plt.title("Логарифмический масштаб")
    plt.show()


N=np.array([5,10,50,100,150])
a,b=0,10

#interpolation(a,b,10,0,0)
interpolation(a,b,100,1,0)
#interpolation(a,b,100,0,1)
#interpolation(a,b,150,1,1)