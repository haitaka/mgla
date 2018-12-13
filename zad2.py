from numpy import linalg as la
from math import *
from matrix import *
import random


N1 = 10
N2 = 20
M = 6
delta = 10**(-3)


def x(i):
    t = i - N1
    if t in range(-N1, 0):
        return - float(t) / float(N1)
    elif t in range(0, N2):
        return float(t) / float(N2)


def f(x):
    return cos(pi * x)


def mat_fun(i, j):
    return x(i) ** j


A = FunMatrix(N2 + N1, M + 1, mat_fun)

b = [f(x(i)) + random.uniform(-1, 1) * delta for i in range(A.i_size)]
npb = np.array(b)

npA = A.forNP()

p0, r_norm, rank, sing = la.lstsq(npA, npb, None)

r = npb - npA.dot(p0)

print("x:", p0)
print()
print("r:", r)


def p(i):
    return sum([p0[k] * x(i)**k for k in range(0, M + 1)])


diff = max([abs(f(x(i)) - p(i)) for i in range(A.j_size)])

print("diff:", diff)

print("r_norm:", sqrt(r_norm))

print("A.cond:", la.cond(npA))

