import numpy as np
from numpy import linalg as la
from matrix import *

N = 10

b = 1./4.

def or_fun(i, j):
    if i == j + 1:
        return -b
    elif i == j - 1:
        return b
    return 0

def m_fun(i, j):
    if (i == N - 1 and j == N) or (j == N - 1 and i == N):
        return 0

    sign = 1 if i < N else -1

    if i == j + 1:
        if i % 2 == 0:
            return sign * b
        else:
            return sign * -b
    elif i == j - 1:
        if j % 2 == 0:
            return sign * b
        else:
            return sign * -b
    return 0

def t_fun(i, j):
    m = [[2, 0, 0], [0, 3, 4], [0, 4, 9]]
    return m[i][j]

m = FunMatrix(N * 2, N * 2, m_fun)
#m = FunMatrix(N, or_fun)
#m = FunMatrix(3, t_fun)

npm = m.forNP()

#print(npm)
#print()

norm = np.linalg.norm(npm, ord=2)

def minus_o(a, b):
    assert a - b != 0
    return a - b

def sturm(lambd):
    def p(j):
        if j == 0:
            return abs(m.get(0, 1)) / minus_o(0, lambd)
        elif j == m.j_size - 1:
            return 1. / (minus_o(minus_o(0, lambd), (m.get(m.j_size - 2, m.j_size - 1) * p(j - 1))))
        else:
            return abs(m.get(j, j+1)) / (minus_o(minus_o(0, lambd), (m.get(j-1, j) * p(j - 1))))

    c = 0
    for i in range(m.i_size):
        if p(i) < 0:
            c += 1
    return c

eigvals, eigvects = la.eig(npm)

for eval, evec in zip(eigvals, np.transpose(eigvects)):
    break
    #print(eval, evec)
    #print("Ax", npm.dot(evec))
    #print("lx", evec * eval)
    print(max(npm.dot(evec) - (evec * eval)))
    print()

print(max([max(npm.dot(evec) - (evec * eval)) for eval, evec in zip(eigvals, np.transpose(eigvects))]))