# -*- coding: utf-8 -*-
"""
author: chris manucredo, chris.manucredo@gmail.com
about:
    this is an implementaion of newton's method for solving systems of 
    non-linear equations. as an example we use two different non-linear
    systems. at the end you'll see a work-accuracy plot for both systems.
"""
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.array([[x[0,0] + 5*x[1,0]**2 - x[1,0]**3 - 2*x[1,0] - 13],[x[0,0] + x[1,0]**2 + x[1,0]**3-14*x[1,0] - 29]]) 

def df(x):
    return np.array([[1, 10*x[1,0] - 3*x[1,0]**2 - 2],[1, 2*x[1,0] + 3*x[1,0]**2 - 14]])

def g(x):
    return np.array([[2*x[0,0] + 3*x[1,0] + 4*x[1,0]**2 - 5*x[1,0]**3 + 16],[-x[0,0] - 2*x[1,0] + 3*x[1,0]**2 - 7*x[1,0]**3 + 49]])

def dg(x):
    return np.array([[2, 3 - 8*x[1,0] - 15*x[1,0]**2],[-1, -2 + 6*x[1,0] - 21*x[1,0]**2]])


"""
usage:
    f: system of non-linear equations
    df: derivative of f
    x0: starting guess
    e: desired tolerance of error
    realSolution: the exact solution of the equation
"""
def newtons_method(f, df, x0, e, realSolution):
    fehler = []
    delta = np.linalg.norm(np.dot(np.linalg.inv(df(x0)),f(x0)))
    while np.linalg.norm(delta) > e:
        x0 = x0 - np.dot(np.linalg.inv(df(x0)),f(x0))
        delta = np.linalg.norm(np.dot(np.linalg.inv(df(x0)),f(x0)))
        fehler.append(np.linalg.norm(realSolution-x0))
    
      
    print('Root is at: ', x0)
    print('f(x) at root is: ', f(x0))
    return fehler
 

   
konv1 = newtons_method(f, df, np.array([[0],[0]]), 10**-16, np.array([[5],[4]]))  
konv2 = newtons_method(g, dg, np.array([[0],[0]]) , 10**-16, np.array([[1],[2]]))

N = np.arange(0,len(konv1),1)
M = np.arange(0,len(konv2),1)

plt.plot(N,konv1)
plt.xlabel("Iterations"); plt.ylabel("Error")
plt.legend(["System 1"])
plt.show()
plt.plot(M,konv2)
plt.xlabel("Iterations"); plt.ylabel("Error")
plt.legend(["System 2"])
plt.show()