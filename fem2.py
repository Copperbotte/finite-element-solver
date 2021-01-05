import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

#fem2 performs a 1d nonlinear finite element analysis.
#m is the global stiffness matrix, initializes to 0.
#L is the length of a single element in the array. fem only solves between 0 and 1.
#the for loop adds the linearization of the stiffness matrix to each pair of terms. This can also be solved numerically.
#m[0][0] is set to 1, as that's the initial condition. The previous for loop skipped the first term, which is filled in here.
#mi is the inverse of the stiffness matrix. This can probably be solved iteratively.
#

#linearization algo:
#take an input approximation, use to calculate A and B coefficients per cell
#use resulting shapefunc matrix to find finite element approximation
#use finite element to re-calculate A and B

def shpfnc(L, a=-1.0, b=1.0, c=0.0):
    #build shapefuncs
    t = sp.symbols('t')#dummy for derivatives
    x = sp.Function('x')(t)
    X = [np.power(x, N) for N in range(2)]
    S = [[_X.subs(x, N) for _X in X] for N in range(2)]
    Si = sp.Matrix(S).inv()

    #find diffeq
    #maybe make this a parameter somehow?
    dX = [sp.diff(i,t) for i in X]
    DE = [d0*a + d1*b + c for d1,d0 in zip(dX, X)]
    R = [sp.integrate(de, (x, 0, 1)) for de in DE]

    #swap derivative with real space derivative
    #L = sp.symbols('L')
    R = [r.subs(sp.Derivative(1, t), sp.diff(t/L, t)) for r in R]

    #evaluate shapefunc residual
    shp = sp.Matrix(R).transpose() * Si
    
    return shp.tolist()[0]

def fem2(n, A=None, B=None, C=None):
    #generate default coefficients
    if not A:
        A = [-1.0] * n
    if not B:
        B = [1.0] * n
    if not C:
        C = [0.0] * n

    #build shapefunc
    L = sp.symbols('L')
    shps = [shpfnc(L, a, b, c) for i,a,b,c in zip(range(n), A, B, C)]
    shps = [[i.subs(L, 1.0/n) for i in shp] for shp in shps]
    
    #build stiffness matrix 
    m = np.array([[0.0 for j in range(n+1)] for i in range(n+1)])
    for i in range(n):
        for s in range(len(shps[0])):
            m[i+1][i+s] = shps[i][s]

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
    #plt.plot(*fem2(2), 'o-', label="2 elements")
    #plt.plot(*fem2(5), 'o-', label="5 elements")
    #plt.plot(*fem2(25), 'o-', label="25 elements")

    #x = np.arange(0, 1+0.001, 0.001)
    #y = np.exp(x)
    #plt.plot(x,y, label="ground truth")
    #plt.legend()
    #plt.show()

    #function and its derivative, for linearization
    def f(x):
        return 1.0 / np.sqrt(1.0 + x**2)
    def df(x):
        return (-x / (1+x**2))*f(x)

    A = None
    B = None
    C = [5.0]*25
    
    for i in range(10):
        x0, y0 = fem2(25, *(A, B, C))
        plt.plot(x0, y0, 'o-', label=str(i), color=str(float(i)/10.0))
        
        #generate A, B, and C coefficients from function
        #should this be another matrix?

        #c should be generated from another parameter somehow
        
        k1 = -1.0
        k2 = 1.0
        c = 5.0

        A = list(map(lambda x: k1*f(x) + k2*x, y0))
        B = list(map(lambda x: k1*df(x), y0))
        C = [c]*len(y0)

    x0, y0 = fem2(25, *(A, B, C))
    plt.plot(x0, y0, 'o-', label=str(10))
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("start")
    plots()
