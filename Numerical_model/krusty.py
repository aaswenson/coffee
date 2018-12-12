#!/usr/bin/python3
import sys
import numpy as np
import time
from units import *
from solvers import runge_kutta
import plot_utilities as pu

"""
PRKE Finite Difference for KRUSTY KRAB REACTOR
"""

class krusty():
    
    # Reactivity temperature coefficient [dk/k/K]
    RTC = lambda self, T:self.matdat.beta*(-7.3e-11*(T)**2\
                                           -7.58e-7*(T) - 1.13e-3)

    # Eq.1 (INL/EXT-10-19373 Thermophysical Properties of U-10Mo Alloy)
    # Range: 100 < T < 1000 [C]
    cp = lambda self, T: 0.137 + 5.12e-5*(T-unit['C_K'])\
                         + 1.99e-8*(T-unit['C_K'])**2

    # Eq.4 (INL/EXT-10-19373 Thermophysical Properties of U-10Mo Alloy)
    # Range: 20 < T < 700 [C]
    dens = lambda self, T: 17.15 - 8.63e-4*(T-unit['C_K']+20)

    # Eq.4 (INL/EXT-10-19373 Thermophysical Properties of U-10Mo Alloy)
    # Range: 20 < T < 800 [C]
    kcond = lambda self, T: 10.2 + 3.51e-2*(T-unit['C_K'])

    # control insertion state, start simulational with step insertion of
    # reactivity so insert == False
    insert = False

    def __init__(self, w_Pu=0):
        self.n0 = 100                     # initial neutron population
        self.rho_cost = 0.60            # Initial step reactivity cost [$]
        self.matdat = FuelMat(w_Pu)
        self.reactor()                  # set Physical parameters of system
        # reactivity insertion
        self.rho_insert = self.rho_cost * self.matdat.beta
 
    def reactor(self):
        self.volume = float(1940)           # Fuel volume [cc]
        # reactor fuel and coolant channels start at room temp
        self.Tf0 = float(293.15)            # Initial Fuel Temperature [K]
        self.T_ref_rho = self.Tf0
        self.mass = self.volume*self.dens(self.Tf0)        # [g/cc]
        
        # heat pipe geometry
        N_ch = 8                            # Number of coolant channels
        D_ch = 0.0107                         # ID of heat pipe
        A_ch = np.pi*D_ch**2/4              # cross-sectional area of heat pipe
        self.A_chs = N_ch * A_ch            # Total cross-sect. area of heat pipes
        r_fi = 0.02                         # inner radius, fuel [m]
        r_fo = 0.055                        # outter radius, fuel [m]
        hgt = 0.25                          # Reactor height [m]
        self.R_f0 = np.log(r_fo/r_fi)/(2*np.pi*hgt) # resitance term

    def RungeKutta4(self):        
        
        times = [10000]
        dts   = [0.002]
        tsec = []
        t0 = 0
        for t, dt in zip(times, dts):
            tsec += list(np.arange(t0, t, dt))
            t0 += t

        c   = np.zeros((len(tsec), self.matdat.groups))
        rho = np.zeros(len(tsec))
        n   = np.zeros(len(tsec))
        Tf  = np.zeros(len(tsec))
        p   = np.zeros(len(tsec))
        dNdt = np.zeros(len(tsec))

        p[0]   = 0
        c[0]   = np.array(self.matdat.c0)
        rho[0] = self.rho_insert
        n[0]   = self.n0
        Tf[0]  = self.Tf0
        
        for ind, t in enumerate(tsec[1:]):
            ind += 1
            dt = t - tsec[ind-1]
            # initial values
            x0 = [n[ind-1], c[ind-1], rho[ind-1], Tf[ind-1]]
            dndt = runge_kutta(x0, self.fn, dt, 0)
            dcdt = runge_kutta(x0, self.fc, dt, 1)
            dTdt = runge_kutta(x0, self.fT, dt, 2)

            n[ind]  = n[ind-1] + dndt 
            c[ind]  = np.add(c[ind-1], dcdt) 
            Tf[ind] = Tf[ind-1] + dTdt
            p[ind]  = n[ind] * self.matdat.n_W
            rho[ind] = self.rho_feedback(Tf[ind], rho[ind])
            dNdt[ind] = dndt    
            str = '{0:.4f} {1:.4e} {2:.4f} {3:.4f}'
            print(str.format(t, p[ind], rho[ind], Tf[ind]))

        pu.save_results(tsec, n, rho, p, Tf, c, dNdt)
    
    def fn(self, args):
        """ Neutron derivative function """
        [n, c, rho, T] = args
        return (rho-self.matdat.beta)/self.matdat.L*n + np.dot(c, self.matdat.lam)

    def fc(self, args):
        """ 6-group precursor derivative function """
        [n, c, rho, T] = args
        
        cs = np.add(np.divide(np.multiply(n, self.matdat.beta_i), self.matdat.L), 
                      np.multiply(-self.matdat.lam, c))
        
        return cs

    def fT(self, args):
        """ Fuel Temperautre function """
        [n, c, rho, T] = args
        
        P_fuel = n*self.matdat.n_W            # Reactor power
        h_bar = 209000                         # [W/m^2-K]
        R_conv = 1/(h_bar*self.A_chs)
        C = self.cp(T) * self.mass
        
        # adiabatic heating
        if T <= 373.15:
            Q_out = 0
        else:
            deltaT = 7
            Q_out = deltaT / R_conv
        
        return (P_fuel - Q_out) / C

    def rho_feedback(self, T, rho1):
        """Calculate reactivity response to temperature.
        """
        if T > 673.15 and self.insert == False:
            self.insert = True
            self.rho_insert = 0
            self.T_ref_rho = 673.15

        return self.rho_insert + self.RTC(T)*(T-self.T_ref_rho)

def main():
    start = time.time()
    kilo = krusty(0.5)
    kilo.RungeKutta4()
    end = time.time()
    print("Calculation time: %0.f" % (end-start))

if __name__ == "__main__":
    main()
