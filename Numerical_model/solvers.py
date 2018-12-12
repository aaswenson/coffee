import numpy as np

def update_value(dxdt, data, dt, ind):
    """
    """
    data[ind] = data[ind-1] + dxdt * dt

def runge_kutta(x0, dX, dt, key):
    """Generalized 4th-order Runge-Kutta Method for estimating time change in
    neutron population, precursor conc, and temperature
    """
    d1 = dX(x0)
    x0[key] = np.add(x0[key], np.multiply(d1, dt/2))

    d2 = dX(x0)
    x0[key] = np.add(x0[key], np.multiply(d2, dt/2))

    d3 = dX(x0)
    x0[key] = np.add(x0[key], np.multiply(d2, dt))

    d4 = dX(x0)
    
    dXdt =  np.add(np.add(np.add(d1, 2*d2), 2*d3), d4)*dt/6

    return dXdt

