#this is a reference for FEM nonlinear, which solves for a nonlinear system.

import numpy as np
import matplotlib.pyplot as plt

def f(n):
    return 1 / np.sqrt(1+n**2)

def df(n):
    return (-n / (1+n**2)) * f(n)

x = np.arange(0,10+0.001,0.001)

def plots(x0=0.5):
    y = list(map(f, x))
    plt.plot(x,y)
    dy = list(map(df, x))
    plt.plot(x,dy)

    #linearize
    a = f(x0)
    b = df(x0)

    ly = a + b*(x - x0)
    plt.plot(x, ly)
    plt.show()
	
plots(3)
