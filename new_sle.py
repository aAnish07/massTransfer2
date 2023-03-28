import numpy as np
from numpy.polynomial.polynomial import polyfit
from sympy import *
from sympy import symbols, Eq, solve
from sympy.abc import x, y
import sympy
from scipy import interpolate

dataP = 50
data = np.ones((dataP, 4))
for w in range(dataP):
    # 1
    yc = [0.0001, 0.05, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    ya = [0.0001, 0.002, 0.005, 0.007, 0.01, 0.013, 0.017, 0.022, 0.029]
    xa = [0.67, 0.66, 0.64, 0.625, 0.60, 0.58, 0.55, 0.51, 0.46]

    xc = np.ones(9)
    xb = np.ones(9)
    yb = np.ones(9)
    mp = np.ones(9)
    slope = np.ones(9)
    for i in range(9):
        yb[i] = 1 - ya[i] - yc[i]
        xc[i] = (yc[i] * (1 - xa[i])) / (yc[i] + yb[i])
        xb[i] = 1 - xa[i] - xc[i]
        mp[i] = (yc[i] + xc[i]) / 2
        slope[i] = (yb[i] - xb[i]) / (yc[i] - xc[i])

    F = 1000
    S = np.random.randint(500, 2500)
   
    yaf = 0.75
    ybf = 0.0
    
    ycfInp = np.random.rand()
    if ycfInp == 0.0 or ycfInp == 1.0:
        ycfInp = 0.5
    ycf = ycfInp
    xcs = 0.0
    xbs = 1.0
    xas = 0.0
    
    f = [ycf, ybf]
    s = [xcs, xbs]

    
    stages = np.random.randint(2, 10)

    L = np.zeros(stages)
    V = np.zeros(stages)
    M = np.zeros(stages)
    Mx = np.zeros(stages)
    My = np.zeros(stages)
    xcl = np.zeros(stages)
    xbl = np.zeros(stages)
    ycv = np.zeros(stages)
    ybv = np.zeros(stages)
    slopeI = np.ones(stages)


    
    # p1 = polyfit(xc, xb, 2)
    # p2 = polyfit(yc, yb, 2)
    
    # p1 = np.polyfit(xc, xb, 2)
    # p2 = np.polyfit(yc, yb, 2)
    # # slopeI = np.ones(stages)
    
    
    # Define the functions to interpolate
    # Note that polyfit returns the coefficients of the polynomial in decreasing order of powers
    p1 = np.polyfit(xc, xb, 2)
    p2 = np.polyfit(yc, yb, 2)
    slope_interp = interpolate.interp1d(mp, slope, kind='linear', fill_value='extrapolate')


    for i in range(stages):
        if i == 0:
            M[i] = F + S
            Mx[i] = (F * ycf + S * xcs) / M[i]
            My[i] = (F * ybf + S * xbs) / M[i]
        elif i > 0:
            M[i] = L[i - 1] + S
            Mx[i] = (L[i - 1] * xcl[i - 1] + S * xcs) / M[i]
            My[i] = (L[i - 1] * xbl[i - 1] + S * xbs) / M[i]

        # Interpolate the slope using the previously defined function
        slopeI[i] = slope_interp(Mx[i])

        # Solve for xcl and xbl
        x, y = sympy.symbols('x y')
        try:
            eq1 = sympy.Eq(y, sympy.poly(p1)(x))
            eq2 = sympy.Eq(y, My[i] + slopeI[i] * (x - Mx[i]))
            xcl[i], xbl[i] = sympy.solve((eq1, eq2), (x, y), ((xc[0], xc[-1]), (xb[-1], xb[0])))[0]
        except:
            continue

        # Solve for ycv and ybv
        x, y = sympy.symbols('x y')
        try:
            eq1 = sympy.Eq(y, sympy.poly(p2)(x))
            eq2 = sympy.Eq(y, My[i] + slopeI[i] * (x - Mx[i]))
            ycv[i], ybv[i] = sympy.solve((eq1, eq2), (x, y), ((yc[0], yc[-1]), (yb[-1], yb[0])))[0]
        except:
            continue

        # Calculate L and V
        L[i] = (M[i] * Mx[i] - M[i] * ycv[i]) / (xcl[i] - ycv[i])
        V[i] = M[i] - L[i]

    fra = (F * ycf - L[1] * xcl[1]) / (F * ycf)

    print(fra)


    
# for i in range(stages):
#     if i == 0:
#         M[i] = F + S
#         Mx[i] = (F*ycf + S*xcs)/M[i]
#         My[i] = (F*ybf + S*xbs)/M[i]
#     else:
#         M[i] = L[i-1] + S
#         Mx[i] = (L[i-1]*xcl[i-1] + S*xcs)/M[i]
#         My[i] = (L[i-1]*xbl[i-1] + S*xbs)/M[i]

#     # interpolation
    
#     slopeI[i] = interp1d(mp, slope, kind='linear')(Mx[i])

#     # solving equations using sympy
#     x, y = symbols('x y')
#     eq1 = Eq(y, poly(p1, x))
#     # eq1 = Eq(y, poly(p1, x))
#     eq2 = Eq(y, My[i] + slopeI[i]*(x-Mx[i]))
#     sol1 = solve([eq1, eq2], [x, y], [(xc[0], xb[-1]), (xb[-1], xc[0])])
#     xcl[i], xbl[i] = sol1[0]
    
#     x, y = symbols('x y')
#     eq1 = Eq(y, poly(p2, x))
#     eq2 = Eq(y, My[i] + slopeI[i]*(x-Mx[i]))
#     sol2 = solve([eq1, eq2], [x, y], [(yc[0], yc[-1]), (yb[-1], yb[0])])
#     ycv[i], ybv[i] = sol2[0]

#     L[i] = (M[i]*Mx[i]-M[i]*ycv[i])/(xcl[i]-ycv[i])
#     V[i] = M[i] - L[i]

    
# M = np.ones(10)
# Mx = np.ones(10)
# My = np.ones(10)
# xbl = np.ones(10)
# xcl = np.ones(10)
# ycv = np.ones(10)
# ybv = np.ones(10)
# L = np.ones(10)
# V = np.ones(10)
# p1 = polyfit(np.array([xcs, 0.5, xbs]), np.array([xcs, 0.5, xbs]), 2)
# p2 = polyfit(np.array([ycf, 0.5, ybf]), np.array([xcs, 0.5, xbs]), 2)

# slopeI = np.ones(10)

# for i in range(stages):
#     if i == 0:
#         M[i] = F + S
#         Mx[i] = (F * ycf + S * xcs) / M[i]
#         My[i] = (F * ybf + S * xbs) / M[i]
#     elif i > 0:
#         M[i] = L[i - 1] + S
#         Mx[i] = (L[i - 1] * xcl[i - 1] + S * xcs) / M[i]
#         My[i] = (L[i - 1] * xbl[i - 1] + S * xbs) / M[i]

#     slopeI[i] = interp1d(mp, slope, kind='linear')(Mx[i])

#     L[i] = (M[i] * Mx[i] - M[i] * ycv[i]) / (xcl[i] - ycv[i])
#     V[i] = M[i] - L[i]

#     xcl[i], xbl[i] = np.roots([p1[0], p1[1] - (My[i] + slopeI[i] * Mx[i]), p1[2] - My[i]])
#     ycv[i], ybv[i] = np.roots([p2[0], p2[1] - (My[i] + slopeI[i] * Mx[i]), p2[2] - My[i]])

# fra = (F * ycf - L[1] * xcl[1]) / (F * ycf)

# data = np.zeros((10, 4))
# data[:, 0] = S
# data[:, 1] = stages
# data[:, 2] = ycfInp