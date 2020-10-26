import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

#fem performs a 1d finite element analysis.
#m is the glopbal stiffness matrix, initializes to 0.
#L is the length of a single element in the array. fem only solves between 0 and 1.
#the for loop adds the linearization of the stiffness matrix to each pair of terms. This can also be solved numerically.
#m[0][0] is set to 1, as that's the initial condition. The previous for loop skipped the first term, which is filled in here.
#mi is the inverse of the stiffness matrix. This can probably be solved iteratively.
#

def shpfnc(L):
    #build shapefuncs
    t = sp.symbols('t')#dummy for derivatives
    x = sp.Function('x')(t)
    X = [np.power(x, N) for N in range(2)]
    S = [[_X.subs(x, N) for _X in X] for N in range(2)]
    Si = sp.Matrix(S).inv()

    #find diffeq
    #maybe make this a parameter somehow?
    dX = [sp.diff(i,t) for i in X]
    DE = [d1 - d0 for d1,d0 in zip(dX, X)]
    R = [sp.integrate(de, (x, 0, 1)) for de in DE]

    #swap derivative with real space derivative
    #L = sp.symbols('L')
    R = [r.subs(sp.Derivative(1, t), sp.diff(t/L, t)) for r in R]

    #evaluate shapefunc residual
    shp = sp.Matrix(R).transpose() * Si
    
    return shp.tolist()[0]

def fem(n):
    #build shapefunc
    L = sp.symbols('L')
    shp = shpfnc(L)
    shp = [i.subs(L, 1.0/n) for i in shp]
    
    #build stiffness matrix 
    m = np.array([[0.0 for j in range(n+1)] for i in range(n+1)])
    for i in range(n):
        for s in range(len(shp)):
            m[i+1][i+s] = shp[s]
            
    #initial condition
    m[0][0] = 1.0

    #find stiffness inverse, to find element heights
    mi = np.linalg.inv(m)
    x = np.array([[1.0 if i == 0 else 0.0 for i in range(n+1)]])
    res = mi.dot(x.transpose())
    x = np.arange(0, 1+1/n, 1/n)

    #returns element positions, and heights
    return x, res.transpose()[0]

def plots():
    plt.plot(*fem(2), 'o-', label="2 elements")
    plt.plot(*fem(5), 'o-', label="5 elements")
    plt.plot(*fem(25), 'o-', label="25 elements")
    x = np.arange(0, 1+0.001, 0.001)
    y = np.exp(x)
    plt.plot(x,y, label="ground truth")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("start")
    plots()
