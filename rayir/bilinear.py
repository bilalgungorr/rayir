import numpy as np
from scipy.optimize import minimize_scalar as scipy_minimize
from scipy.optimize import root_scalar as scipy_root_scalar
from scipy.optimize import fmin as scipy_fmin

def interpolate(x2, x0, x1, y0, y1):
    y2 = y1 + (x2 - x1)*(y1 - y0)/(x1 - x0)
    return y2

def findex(x, array):
    if x <= array[0]:
        return 0
    for i in range(len(array) - 1):
        x1, x2 = array[i: i+2]
        if x1 <= x <= x2:
            return i
    else:
        return array.size - 1

def find_value(x, xarray, yarray):
    for i in range(len(xarray) - 1):
        x1, x2 = xarray[i: i+2]
        if x1 <= x <= x2 or x1 >= x >= x2:
            s1 = slice(i, i+2)
            return interpolate(x, *xarray[s1], *yarray[s1])

def find_value2(x, xarray, yarray):
    if x <= min(xarray):
        return interpolate(x, *xarray[0: 2], *yarray[0: 2])
    elif x >= max(xarray):
        return interpolate(x, *xarray[-2: ], *yarray[-2: ])
    return find_value(x, xarray, yarray)


def bilinear_response(curvature, moment, method='UTC-40'):
    """
    Return (ndarray): np.array([[0, phiY, phiU], [0, My, Mu]])
        bilinear represantation of moment-curvature relation.
        
    Parameters:
        method (str): {'UTC-40', 'regression'}
    """

    method = method.upper()

    if not method in ['UTC-40', 'REGRESSION']:
        raise ValueError("method must be one of {'UTC-40', 'regression'}")

    if method.upper() == 'UTC-40':
        phiy, my = _find_yield_point(curvature, moment)
        return np.array([
            [0, phiy, curvature[-1]], [0, my, moment[-1]]
            ])
    else:
        return _bilinear_with_point(curvature, moment)


def _bilinear_with_point(xarray, yarray):
    rss_sum = 0
    RSS = []
    size = xarray.size
    x2 = xarray[-1]

    def function(x1, return_rss):
        i1 = findex(x1, xarray) + 1

        xarr2 = xarray[i1:]
        yarr2 = yarray[i1:]

        A = np.vstack([xarr2, np.ones(size-i1)]).T
        [m2, c2], rss2, *_ = np.linalg.lstsq(A, yarr2, rcond=None)

        m1 = (m2*x1 + c2)/x1

        xarr1 = xarray[:i1+1]
        yarr1 = yarray[:i1+1]
        rss1 = ((m1*xarr1 - yarr1)**2).sum()

        m1 = (m2*x1 + c2)/x1
        y1 = m1*x1
        y2 = m2*x2 + c2


        if return_rss:
            if rss2.size > 0:
                return rss1 + rss2[0]
            return rss1
        return x1, m2, c2

    res = scipy_minimize(
            function, bounds=(1e-5, xarray.mean()), 
            args=(True,), method='bounded')

    x1, m2, c2 = function(res.x, False)
    
    m1 = (m2*x1 + c2)/x1
    y1 = m1*x1
    y2 = m2*x2 + c2


    return np.array([[0, x1, x2], [0, y1, y2]])


    if keff is None:
        ke0 = np.nanmin(forces[1:5]/deformations[1:5])


        sol = scipy_root_scalar(function, method='secant',
                bracket=[0.1*ke0, ke0*2],
                x0=ke0*0.5, x1=ke0)

        ke = sol.root

    else:
        ke = keff
    
    function(ke)
    return dy_try, fy_try 


def _bilinear_with_index(xarray, yarray):
    rss_sum = 0
    RSS = []
    size = xarray.size

    x2 = xarray[-1]
    for i in range(2, size//3, 1):
        xi = xarray[i]

        xarr2 = xarray[i:]
        yarr2 = yarray[i:]

        A = np.vstack([xarr2, np.ones(size-i)]).T
        [m2, c2], rss2, *_ = np.linalg.lstsq(A, yarr2, rcond=None)

        m1 = (m2*xi + c2)/xi

        xarr1 = xarray[:i+1]
        yarr1 = yarray[:i+1]
        rss1 = ((m1*xarr1 - yarr1)**2).sum()

        if rss2.size > 0:
            RSS.append([i, rss1 + rss2[0]])
        else:
            RSS.append([i, rss1])

        y1 = m1*xi
        y2 = m2*x2 + c2


    i = min(RSS, key=lambda x: x[1])[0]
    x1 = xarray[i]

    xarr2 = xarray[i:]
    yarr2 = yarray[i:]

    A = np.vstack([xarr2, np.ones(size-i)]).T
    [m2, c2], rss2, *_ = np.linalg.lstsq(A, yarr2, rcond=None)

    m1 = (m2*x1 + c2)/x1
    y1 = m1*x1
    y2 = m2*x2 + c2

    return np.array([[0, x1, x2], [0, y1, y2]])


def _find_yield_point(deformations, forces, keff=None):
    len_data = len(deformations)
    du = deformations[-1]
    fu = forces[-1]
    
    # area under capacity curve
    area_curve = 0.
    for i in range(len_data - 1):
        d1, d2 = deformations[i: i+2]
        f1, f2 = forces[i: i+2]
        area_curve += 0.5*(f1 + f2)*(d2 - d1)

    def area_bil_func(dy, ke):
        """Compute area under bilinear curve"""
        fy = ke*dy
        return 0.5*((fu + fy)*(du - dy) + fy*dy)
    

    # for binding
    dy_try, fy_try = 0, 0

    def function(ke):
        nonlocal dy_try, fy_try
        
        # initial deformation
        d0 = deformations[len_data//5]
        d1 = 1.2*d0
        # initial areas based on initial deformation and stiffness
        a0, a1 = area_bil_func(d0, ke), area_bil_func(d1, ke)
        
        # deformation that makes area under bilinear curve equal to area_curve
        dy_try = interpolate(area_curve, a0, a1, d0, d1)
        #print(area_curve, area_bil_func(dy_try, ke))

        fy_try = ke*dy_try
        
        # finding deformation at which corresponding force 
        # equal to 60 % of yield force (fy_try)
        dy06 = find_value(0.6*fy_try, forces, deformations)
        fy06 = ke*dy06
        
        return 0.6*dy_try - dy06

    if keff is None:
        ke0 = np.nanmin(forces[1:5]/deformations[1:5])

        sol = scipy_root_scalar(function, method='secant',
                bracket=[0.1*ke0, ke0*2],
                x0=ke0*0.5, x1=ke0)
        ke = sol.root

    else:
        ke = keff
    
    function(ke)
    return dy_try, fy_try 

