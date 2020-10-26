import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#This is a sample finite element solver from page 8 of "Introduction to the Finite element method" by G. P. Nikishkov.
def fem_sample(n=3):
    #make shapes
    x = sp.symbols('x')
    approc = [x**i for i in range(2)]
    points = [[element.subs(x, float(i)) for element in approc] for i in range(2)]
    shapes = np.linalg.inv(np.array(points).astype(np.float64))
    shapes = np.array(approc).dot(shapes)
    print(shapes)

    #galerkin on each pair of nodes
    #i don't know how to mark substitutable derivatives in sympy, so this is hardcoded
    #y" + c = 0
    shapes = np.array([shapes])
    st = shapes.transpose()
    ds = np.array([sp.diff(e, x) for e in shapes])
    dst = ds.transpose()
    diffeq = st.dot(dst) - st.dot(shapes)
    diffeq = [[sp.expand(e) for e in row] for row in diffeq]
    iinte = [[sp.integrate(e) for e in row] for row in diffeq]
    dinte = [[e.subs(x,1) - e.subs(x,0) for e in row] for row in iinte]
    print(dinte)
    
    #build stiffness matrix
    M = np.zeros((n,n))
    for i in range(n-1):
        for u in range(2):
            for v in range(2):
                M[u+i][v+i] += dinte[u][v]
    print(M)
    for i in range(1,n):
        M[0][i] = 0
        M[i][0] = 0
    M[0][0] = 1
    print(M)

    L = 2.0 / float(n-1)
    M /= L
    
    Bv = np.array([2.0 for i in range(n)])
    Bv[0] = 0.0
    Bv[-1] = 1.0
    Bv *= L / 2.0
    Bv[-1] += 1.0
    Bv = np.transpose(Bv)
    Mi = np.linalg.inv(M)
    U = Mi.dot(Bv)
    x = np.arange(len(U)) / float(len(U) - 1)
    plt.plot(x,U)

#this is a generalization of the above solver, using its matrix solution.
#generalizes to arbitrarily large numbers of samples.
def sample(n=3):
    L = 2.0 / float(n-1)
    M = np.zeros((n,n))
    F = np.array([[1,-1],[-1,1]]) # inverse shape functions
    for i in range(n-1):
        for x in range(2):
            for y in range(2):
                M[x+i][y+i] += F[x][y]
    for i in range(1,n):
        M[0][i] = 0
        M[i][0] = 0
    M[0][0] = 1
    print(M)
    M /= L

    Bv = np.array([2.0 for i in range(n)])
    Bv[0] = 0.0
    Bv[-1] = 1.0
    Bv *= L / 2.0
    Bv[-1] += 1.0
    Bv = np.transpose(Bv)
    Mi = np.linalg.inv(M)
    U = Mi.dot(Bv)
    x = np.arange(len(U)) / float(len(U) - 1)
    plt.plot(x,U)

def plots():
    sample(100)
    sample(3)
    sample(2)
    plt.show()

if __name__ == "__main__":
    plots()

#this is an attempt at deriving the above solver from fundamental principles.
#fem.py replaces this function
def fem(n=3):
    #make shapes
    x = sp.symbols('x')
    approc = [x**i for i in range(2)]
    points = [[element.subs(x, float(i)) for element in approc] for i in range(2)]
    shapes = np.linalg.inv(np.array(points).astype(np.float64))
    shapes = np.array(approc).dot(shapes)
    print(shapes)

    #galerkin on each pair of nodes
    #i don't know how to mark substitutable derivatives in sympy, so this is hardcoded
    #y' - y = 0
    shapes = np.array([shapes])
    st = shapes.transpose()
    ds = np.array([sp.diff(e, x) for e in shapes])
    diffeq = st.dot(ds) - st.dot(shapes)
    diffeq = [[sp.expand(e) for e in row] for row in diffeq]
    iinte = [[sp.integrate(e) for e in row] for row in diffeq]
    dinte = [[e.subs(x,1) - e.subs(x,0) for e in row] for row in iinte]

    #build stiffness matrix
    M = np.zeros((n,n))
    for i in range(n-1):
            for u in range(2):
                    for v in range(2):
                            M[u+i][v+i] += dinte[u][v]
    #print(M)
    #for i in range(1,n):
            #M[0][i] = 0
            #M[i][0] = 0
    #M[0][0] = 1
    print(M)
    Mi = np.linalg.inv(M)
    print(Mi)
    return Mi
