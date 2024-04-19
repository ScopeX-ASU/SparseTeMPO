import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def func(x, a):
    return np.exp(a * x)


# def func(x, a, b, c, d, e):
#     return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + 1

## used to fit polynomial
# xdata = np.array([7, 8, 9, 10, 12, 15, 20])
# ydata = np.array([0.23813, 0.17764, 0.13674, 0.10934, 0.07203, 0.04148, 0.0134])

## used to fit exponential
xdata = np.array([15, 20, 25])
ydata = np.array([0.04148, 0.0134, 0.0045])
popt, pcov = curve_fit(func, xdata, ydata)
print(popt, pcov)
ypred = func(xdata, *popt)
print(r2_score(ydata, ypred))
print(ypred)
print(ydata)

xdata = np.linspace(0, 30, 100)
plt.plot(xdata, func(xdata, *popt), "b-", label="data")
plt.savefig("./unitest/figs/crosstalk_fit.png")
