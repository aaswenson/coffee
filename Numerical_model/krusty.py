#!/usr/bin/python3
import sys
import os
import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
from units import *

"""
PRKE Finite Difference for KRUSTY KRAB REACTOR
"""

class krusty():
    # control insertion state, start simulational with step insertion of
    # reactivity so insert == False
    insert = False
    def __init__(self, w_Pu=0):
        self.n0 = 1                     # initial neutron population
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
        
        times = [300, 1800]
        dts   = [0.001, 0.001]
        tsec = []
        t0 = 0
        for t, dt in zip(times, dts):
            tsec += list(np.arange(t0, t, dt))
            t0 += t

        c   = np.zeros((len(tsec), self.matdat.groups))
        rho = np.zeros(len(tsec))
        n   = np.zeros(len(tsec))
        Tf  = np.zeros(len(tsec))
        Tc  = np.zeros(len(tsec))
        p   = np.zeros(len(tsec))

        p[0]   = 0
        c[0]   = np.array(self.matdat.c0)
        rho[0] = self.rho_insert
        n[0]   = self.n0
        Tf[0]  = self.Tf0
        Tc[0]  = self.Tf0
        
            

        for ind, t in enumerate(tsec[1:]):
            ind += 1
            dt = t - tsec[ind-1]
            # initial values
            n_j0 = n[ind-1]
            c_j0 = c[ind-1]
            T_j0 = Tf[ind-1]
            Tcj0 = Tc[ind-1]
            rho_j = self.rho_feedback(T_j0, rho[ind-1])
            
            # first derivative
            dna = self.fn(n_j0, c_j0, rho_j)
            dca = self.fc(n_j0, c_j0, rho_j)
            dTa = self.fTemp(n_j0, T_j0)
            #
            n_j1 = n_j0 + dna*dt/2
            c_j1 = np.add(c_j0, np.multiply(dca, dt/2))
            T_j1 = T_j0 + dTa*dt/2
            #
            dnb = self.fn(n_j1, c_j1, rho_j)
            dcb = self.fc(n_j1, c_j1, rho_j)
            dTb = self.fTemp(n_j1, T_j1)
            #
            n_j2 = n_j0 + dnb*dt/2
            c_j2 = np.add(c_j0, np.multiply(dcb, dt/2))
            T_j2 = T_j0 + dTb*dt/2
            #
            dnc = self.fn(n_j2, c_j2, rho_j)
            dcc = self.fc(n_j2, c_j2, rho_j)
            dTc = self.fTemp(n_j2, T_j2)
            #
            n_j3 = n_j0 + dnc*dt
            c_j3 = np.add(c_j0, np.multiply(dcc, dt))
            T_j3 = T_j0 + dTc*dt
            #
            dnd = self.fn(n_j3, c_j3, rho_j)
            dcd = self.fc(n_j3, c_j3, rho_j)
            dTd = self.fTemp(n_j3, T_j3)
            #
            dndt =  (dna + 2*dnb + 2*dnc + dnd)*dt/6
            dcdt =  np.add(np.add(np.add(dca, 2*dcb), 2*dcc), dcd)*dt/6
            dTdt =  (dTa + 2*dTb + 2*dTc + dTd)*dt/6
            
            n_1 = n_j0 + (dna + 2*dnb + 2*dnc + dnd)*dt/6
            c_1 = c_j0 + np.add(np.add(np.add(dca, 2*dcb), 2*dcc), dcd)*dt/6
            T_1 = T_j0 + (dTa + 2*dTb + 2*dTc + dTd)*dt/6
            
            n[ind]  = n_1
            Tf[ind] = T_1
            p[ind]  = n_1 * self.matdat.n_W
            rho[ind] = rho_j
            c[ind] = np.array(c_1)
            
#            if ind % 5000 == 0 and ind > 0:
#                self.plot_results(tsec[:ind],
#                                  n[:ind],
#                                  rho[:ind],
#                                  Tf[:ind],
#                                  c[:ind])
            
            str = '{0:.4f} {1:.4e} {2:.4f} {3:.4f}'
            print(str.format(t, p[ind], rho[ind], T_1))

        self.plot_results(tsec,n,rho,Tf,c)
    
    def fn(self, n, c, rho):
        """ Neutron derivative function """
        return (rho-self.matdat.beta)/self.matdat.L*n + np.dot(c, self.matdat.lam)

    def fc(self, n, c, rho):
        """ 6-group precursor derivative function """
        
        cs = np.add(np.divide(np.multiply(n, self.matdat.beta_i), self.matdat.L), 
                      np.multiply(-self.matdat.lam, c))
        
        return cs

    def fci(self, group, n, c, rho):
        """ Single group (of 6) precursor derivative function """
        return self.matdat.beta_i[group] / self.matdat.L * n - self.matdat.lam[group] * c

    def fTemp(self, n, T):
        """ Fuel Temperautre function """
        P_fuel = n*self.matdat.n_W            # Reactor power
        deltaT = T/20                         # Temp. diff. from fuel to coolant(bulk)
        h_bar = 20000                         # [W/m^2-K]
        R_f = self.R_f0/self.kcond(T)
        R_conv = 1/(h_bar*self.A_chs)
        R_total = R_f + R_conv
        
        C = self.cp(T) * self.mass
        
        if T <= 373.15:
            Q_out = 0
        else:
            deltaT = 373
            Q_out = deltaT/R_total
        
        return (P_fuel - Q_out)/C

    def rho_feedback(self,T, rho1):
        """Calculate reactivity response to temperature.
        """
        if T > 623.15 and self.insert == False:
            self.insert = True
            self.rho_insert = rho1
            self.T_ref_rho = 623.15

        return self.rho_insert + self.RTC(T)*(T-self.T_ref_rho)

    def RTC(self, T):
        """ Fuel Temperautre Reactivity Coeficient [K^-1] """
        alpha = self.matdat.beta*(-7.3e-9*(T)**2 -7.58e-5*(T) -0.113)/100
        return alpha

    def cp(self, T):
        """ Fuel Specific Heat [J/g-C] """
        # Eq.1 (INL/EXT-10-19373 Thermophysical Properties of U-10Mo Alloy)
        # Range: 100 < T < 1000 [C]
        T = T - unit['C_K']

        return float(0.137 + 5.12e-5*T + 1.99e-8*T**2) 

    def dens(self, T):
        """ Fuel Density [g/cc] """
        # Eq.4 (INL/EXT-10-19373 Thermophysical Properties of U-10Mo Alloy)
        # Range: 20 < T < 700 [C]
        return float(17.15 - 8.63e-4*(T+20))

    def kcond(self, T):
        """ Fuel Thermal conductivity [W/m-C] """
        # Eq.4 (INL/EXT-10-19373 Thermophysical Properties of U-10Mo Alloy)
        # Range: 20 < T < 800 [C]
        return float(10.2 + 3.51e-2*T)
        # Haynes-230    return (float(0.02*T + 8.4315))
    
    def plot_results(self, time, n, rho, Tf, c):
        # Neutron population
        plt.figure()
        plt.plot(time,n)
        plt.title('Neutron Density')
        plt.xlabel('Time [sec]')
        plt.ylabel('n(t)')
        plt.yscale('log')
        plt.savefig('npop.png')
        # Reactivity
        plt.figure()
        plt.plot(time,rho)
        plt.title('Reactivity')
        plt.xlabel('Time [sec]')
        plt.ylabel(r'$\rho(t)$')
        plt.yscale('linear')
        plt.savefig('rho.png')
        # Fuel Temperature
        plt.figure()
        plt.plot(time,Tf, label='Fuel Temp')
        plt.title('Fuel Temperature')
        plt.xlabel('Time [sec]')
        plt.ylabel(r'T_f(t)')
        plt.yscale('linear')
        plt.savefig('temp.png')
        # Reactor Power
        plt.figure()
        power = np.multiply(n,self.matdat.n_W)
        plt.plot(time,power,label = 'watts')
        plt.xlabel('Time [sec]')
        plt.ylabel('P(t)')
        plt.yscale('log')
        plt.savefig('power.png')
        
        plt.clf()
        plt.close()

def main():
    start = time.time()
    kilo = krusty()
    # kilo.Heun()
    kilo.RungeKutta4()
    end = time.time()
    print("Calculation time: %0.f" % (end-start))


if __name__ == "__main__":
    main()
