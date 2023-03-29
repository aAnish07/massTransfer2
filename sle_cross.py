import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def cross_sle(ycf, S, stages):

    yc = np.array([0.0001, 0.05, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    ya = np.array([0.0001, 0.002, 0.005, 0.007, 0.01, 0.013, 0.017, 0.022, 0.029])
    xa = np.array([0.67, 0.66, 0.64, 0.625, 0.60, 0.58, 0.55, 0.51, 0.46])


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
        plt.plot([xc[i], yc[i]], [xb[i], yb[i]])
        # plt.hold(True)
        plt.grid(True)

    plt.plot(xc, xb)
    # plt.hold(True)
    plt.plot(yc, yb)
    # plt.hold(True)

    F = 1000
    # S = 1500
    # S = float(input("Enter your value : "))

    yaf = 0.75
    ybf = 0.0
    xcs = 0.0
    xbs = 1.0
    xas = 0.0
    # ycf = 0.25
    # ycf = float(input("Enter your value : "))

    # stages = 2
    # stages = int(input("Enter your value : "))

    M = np.ones(10)
    Mx = np.ones(10)
    My = np.ones(10)
    xbl = np.ones(10)
    xcl = np.ones(10)
    ycv = np.ones(10)
    ybv = np.ones(10)
    L = np.ones(10)
    V = np.ones(10)
    slopeI = np.ones(10)
    
    
    plt.figure(1)
    f = [ycf, ybf]
    s = [xcs, xbs]
    plt.plot([f[0], s[0]], [f[1], s[1]])
    plt.text(ycf, ybf, '    F')
    plt.text(xcs, xbs, '     S')
    # plt.hold(True)

    p1 = np.polyfit(xc, xb, 2)
    p2 = np.polyfit(yc, yb, 2)

    for i in range(stages):
        if i == 0:
            M[i] = F + S
            Mx[i] = (F * ycf + S * xcs) / M[i]
            My[i] = (F * ybf + S * xbs) / M[i]
        else:
            M[i] = L[i - 1] + S
            Mx[i] = (L[i - 1] * xcl[i - 1] + S * xcs) / M[i]
            My[i] = (L[i - 1] * xbl[i - 1] + S * xbs) / M[i]

        slopeI[i] = np.interp(Mx[i], mp, slope)

        def equations(vars):
            x, y = vars
            return (y - np.polyval(p1, x), y - My[i] - slopeI[i] * (x - Mx[i]))

        try:
            xcl[i], xbl[i] = fsolve(equations, (xc[0], xb[-1]))
        except:
            continue

        def equations(vars):
            x, y = vars
            return (y - np.polyval(p2, x), y - My[i] - slopeI[i] * (x - Mx[i]))

        try:
            ycv[i], ybv[i] = fsolve(equations, (yc[0], yb[-1]))
        except:
            continue

        L[i] = (M[i]*Mx[i] - M[i]*ycv[i])/(xcl[i] - ycv[i])
        V[i] = M[i] - L[i]

    fra = (F*ycf - L[stages-1]*xcl[stages-1])/(F*ycf)
    
    for i in range(stages):
        plt.plot([xcl[i], ycv[i]], [xbl[i], ybv[i]], 'go-')
        txt2 = 'L' + str(i+1)
        plt.text(xcl[i], xbl[i], txt2)
        plt.plot(xcl[i], xbl[i], 'o')
        txt3 = 'V' + str(i+1)
        plt.text(ycv[i], ybv[i], txt3)
        plt.plot(ycv[i], ybv[i], 'o')
        txt4 = 'M' + str(i+1)
        plt.text(Mx[i], My[i], txt4)
        plt.plot(Mx[i], My[i], 'o')
        
        
    plt.xlabel('xc, yc')
    plt.ylabel('xb, yb')
    plt.title("SLE Right Angle Ternary Phase Diagram",loc='center')
    plt.show()

    plt.figure(2)
    stages = int(stages+1)
    arr = list(range(1, stages))
    plt.plot(arr, ycv[0:stages-1],label='overflow')
    plt.plot(arr, xcl[0:stages-1],label='underflow')
    plt.legend(loc='upper right')
    plt.title('overflow and underflow concentrations')
   
    # plt.legend(['underflow'])
    plt.show()
    return fra




#cross_sle(0.25, 2500, 5)
# cross_sle(0.25, 1500, 5)
cross_sle(0.5, 2500, 5)