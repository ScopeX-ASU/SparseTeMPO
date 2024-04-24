'''
Date: 2024-04-23 02:19:26
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-23 02:19:26
FilePath: /SparseTeMPO/unitest/test_crosstalk_fit.py
'''
"""
Date: 2024-04-22 12:08:46
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-22 17:54:07
FilePath: /SparseTeMPO/unitest/test_crosstalk_fit.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from core.models.layers.utils import polynomial

import torch


def exp_func(x, a):
    return np.exp(a * x)


def func(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + np.exp(-f * x)


def exp_func(x, a, b):
    return a * np.exp(b * x)


## used to fit polynomial
# xdata = np.array([7, 8, 9, 10, 12, 15, 20])
# ydata = np.array([0.23813, 0.17764, 0.13674, 0.10934, 0.07203, 0.04148, 0.0134])

xdata = np.array([7, 8, 9, 10, 12, 15, 18, 20, 23, 24, 25])
ydata = np.array(
    [
        0.22030956,
        0.16927033,
        0.12758723,
        0.10066663,
        0.06864623,
        0.03907187,
        0.02318678,
        0.01727944,
        0.01151909,
        0.01027955,
        0.00897099,
    ]
)


def poly_func(x, a, b, c, d, e):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + 1


popt_poly, pcov = curve_fit(poly_func, xdata[:9], ydata[:9])
print(popt_poly, pcov)
ypred1 = poly_func(xdata[:9], *popt_poly)
print(r2_score(ydata[:9], ypred1))
print(ypred1)
print(ydata[:9])
# print(polynomial(torch.from_numpy(xdata[:9]).float(), torch.tensor(popt_poly.tolist() + [1])))
# print(popt_poly[0]*(xdata[:9]**5) + popt_poly[1]*(xdata[:9]**4) + popt_poly[2]*(xdata[:9]**3) + popt_poly[3]*(xdata[:9]**2) + popt_poly[4]*xdata[:9] + 1)
# coeff = torch.tensor(popt_poly)
# xdata_th = torch.from_numpy(xdata[:9])
# print(coeff[0]*xdata_th**5+coeff[1]*xdata_th**4+coeff[2]*xdata_th**3+coeff[3]*xdata_th**2+coeff[4]*xdata_th+1)

popt_exp, pcov = curve_fit(exp_func, xdata[8:], ydata[8:])
print(popt_exp, pcov)
ypred2 = exp_func(xdata[8:], *popt_exp)
print(r2_score(ydata[8:], ypred2))
print(ypred2)
print(ydata[8:])


def func(x, popt_poly, popt_exp):
    return np.where(x < 23, poly_func(x, *popt_poly), exp_func(x, *popt_exp))


xdata1 = np.linspace(0, 30, 100)
plt.plot(xdata1, func(xdata1, popt_poly, popt_exp), "b-", label="data")
plt.plot(xdata, ydata, "ro", label="data")
plt.savefig("./unitest/figs/crosstalk_fit.png", dpi=300)


# ## used to fit exponential
# xdata = np.array([15, 20, 25])
# ydata = np.array([0.04148, 0.0134, 0.0045])
# popt, pcov = curve_fit(func, xdata, ydata)
# print(popt, pcov)
# ypred = func(xdata, *popt)
# print(r2_score(ydata, ypred))
# print(ypred)
# print(ydata)

# xdata = np.linspace(0, 30, 100)
# plt.plot(xdata, func(xdata, *popt), "b-", label="data")
# plt.savefig("./unitest/figs/crosstalk_fit.png")
