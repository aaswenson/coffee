#!/usr/bin/python3
import sys
import numpy as np
import time
from units import *
from solvers import runge_kutta, update_value
import plot_utilities as pu

"""
PRKE Finite Difference for KRUSTY KRAB REACTOR
"""

class krusty():
    
    rho_follow = False
    
    # Reactivity temperature coefficient [dk/k/K]
    RTC = lambda self, T: self.matdat.beta*(-7.3e-11*(T)**2 
                                            -7.58e-7*(T) 
                                            - 1.13e-3)

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
        self.matdat = FuelMat(w_Pu)
        self.reactor()                  # set Physical parameters of system
        self.n0 = 100                   # initial neutron population
        self.rho_func = self.rho_free
        self.rho_cost = 0.15            # Initial step reactivity cost [$]
        self.stop_follow = 0.30*self.matdat.beta
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
        
        times = [2000]
        dts   = [0.001]
        tsec = []
        t0 = 0
        for t, dt in zip(times, dts):
            tsec += list(np.arange(t0, t, dt))
            t0 += t

        tp = np.dtype([('times', 'f8'), ('npop', 'f8'), ('rho', 'f8'),
                       ('Tf', 'f8'), ('c', 'f8', (6,)), 
                       ('power', 'f8'), ('dndt', 'f8'), ('rho_insert', 'f8'), 
                       ('rho_feedback', 'f8')])

        data = np.zeros(len(tsec), dtype=tp)
        
        data['times']    = tsec
        data['c'][:]     = self.matdat.c0*self.n0
        data['rho'][:]   = self.rho_insert
        data['npop'][:]  = self.n0
        data['Tf'][:]    = self.Tf0
        
        for ind, t in enumerate(data[1:]['times']):
            ind += 1
            dt = t - data[ind-1]['times']
            # initial values
            x0 = data[ind-1]
            dndt = runge_kutta(x0, self.fn, dt, 'npop')
            dcdt = runge_kutta(x0, self.fc, dt, 'c')
            dTdt = runge_kutta(x0, self.fT, dt, 'Tf')
            
            update_value(dndt, data['npop'], dt, ind)
            update_value(dcdt, data['c'], dt, ind)
            update_value(dTdt, data['Tf'], dt, ind)
            
            data[ind]['power'] = data[ind]['npop']*self.matdat.n_W / dt
            data[ind]['dndt'] = dndt    
            self.rho_func(data[ind])
            str = '{0:.4f} {1:.4e} {2:.4e} {3:.4f}'
            dat = data[ind]
            print(str.format(t, dat['npop'], dat['rho'], dat['rho_feedback']))
            

        pu.save_results(data)
         
    def fn(self, data):
        """ Neutron derivative function """
        
        c = data['c']
        n = data['npop']
        rho = data['rho']

        dndt =  ((rho-self.matdat.beta)/self.matdat.L)*n + np.dot(c, self.matdat.lam)
        
        return dndt

    def fc(self, data):
        """ 6-group precursor derivative function """
        
        c   = data['c']
        n   = data['npop']
        
        dcdt = np.add(np.divide(np.multiply(n, self.matdat.beta_i), self.matdat.L), 
                      np.multiply(-self.matdat.lam, c))
        
        return dcdt

    def fT(self, data):
        """ Fuel Temperautre function """
        
        n = data['npop']
        p = data['power']
        T = data['Tf']        
        C = self.cp(T) * self.mass

        # adiabatic heating
        if T <= 373.15:
            Q_out = 0
        else:
            Q_out = 0 #3000

        dTdt = (p - Q_out) / C
        
        return dTdt
    
    def rho_free(self, data):
        """Reactivity for free run
        """
        
        T = data['Tf']
        data['rho_feedback'] = self.RTC(T)*(T-self.T_ref_rho)
        data['rho_insert'] = self.rho_insert
        data['rho'] = data['rho_insert'] + data['rho_feedback']
        

    def rho_maintain(self, data):
        """Calculate reactivity response to temperature.
        """
        T = data['Tf']
        data['rho_feedback'] = self.RTC(T)*(T-self.T_ref_rho)
        
        if (-data['rho_feedback'] > self.rho_insert) and\
           (self.rho_follow == False):
            self.rho_follow = True
        
        if self.rho_follow and (self.rho_insert < self.stop_follow):
            self.rho_insert = -data['rho_feedback']
        
        data['rho_insert'] = self.rho_insert
        data['rho'] = data['rho_insert'] + data['rho_feedback']
        
def main():
    start = time.time()
    kilo = krusty(1)
    kilo.RungeKutta4()
    end = time.time()
    print("Calculation time: %0.f" % (end-start))

if __name__ == "__main__":
    main()
