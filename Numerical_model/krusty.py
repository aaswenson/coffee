#!/usr/bin/python3
import sys
import os
import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
"""
PRKE Finite Difference for KRUSTY KRAB REACTOR
"""
__author__ = "Daniel Cech"

class krusty():
    def __init__(self):
        self.Tf = []
        self.rho = []
        # TODO: find better values for L and nO
        self.L = float(0.00002)
        self.n0 = float(1e5)
        self.n = [self.n0]

        # D. Hetrick, Table 1-2, p.11
        # Decay constant, lambda [sec^-1]
        self.lam = [0.0127, 0.0317, 0.115, 0.311, 1.40, 3.87]
        # Neutrons per fission, U-235 Fast Fission, nu
        self.nu_235 = float(2.57)
        # Delayed Neutron Yield [n/fission]
        self.beta_nu = [0.00063, 0.00351, 0.00310, 0.00672, 0.00211, 0.00043]
        # Delayed Neutron Fractions
        self.beta_i = np.divide(self.beta_nu,self.nu_235)
        # Total Beta Fraction
        self.beta = sum(self.beta_i)
        # Number of delayed groups
        self.groups = len(self.lam)
        # Initial Precursor Vectors
        self.c = [[beta*self.n0/(self.L*lam)] for beta,lam in zip (self.beta_i, self.lam)]
    
    def setInputs(self):
        Power_rated = float(5e3)            # Rated thermal power
        Tf_ave = float(800)                 # deg C
        Tc_ave = float(750)                 # deg C
        M_235 = (27.5e3)                    # grams U-235
        phi_ave = float(9.3e11)             # Core Average Neutron Flux
        nom_Wcc = float(1.6)                # Nominal Power density

    def Heun(self):
        rho_cost = float(0.6)              # Reactivity [$]
        rho0 = rho_cost*self.beta           # Reactivity [delta(k)/k]
        alpha = 0.15                        # Fuel reactivity temperature coefficient
        K =0.017                            # [C/kw-sec]
        Tf0 = float(20)                     # Initial Fuel Temperature [C]
        Ttime = float(100)               # total time [3 hours]
        Nstep = float(10000)                  # total steps
        dt = Ttime/Nstep                    # time step size
        tsec = np.arange(0,Ttime+dt,dt)
        rho = [rho0]
        Tf = [Tf0]
        n = []
        c = []
        n = [self.n0]
        c = self.c
        for ind,tm in enumerate(tsec[1:]):
            clast = [x[-1] for x in c]                          # precursors from last time step
            n_pred = n[-1] + dt*self.fn(n[-1],clast,rho[-1])    # predictor step
            c_pred = ([cl + dt*f for cl,f in zip(clast,self.fc(n[-1],clast,rho[-1]))])
            rho = [rho0 - alpha*(Tf[-1]-Tf0)]

            n_corr = n[-1] + dt/2*(self.fn(n[-1],clast,rho[-1])+self.fn(n_pred,c_pred,rho[-1]))
            n.append(n_corr)
            for i in range(self.groups):
                c[i].append(clast[i]+dt/2*(self.fci(i,n[-1],clast[i],rho[-1]) + self.fci(i, n_pred, c_pred[i], rho[-1])))
        print('done')
        plt.plot(tsec,n,label=rho_cost)
        plt.ylim(-1e8,1e8)
        n = 0
        plt.legend()
        plt.show()

    def fn(self,n,c,rho):
        return (rho-self.beta)/self.L*n + np.dot(c,self.lam)
    
    def fc(self,n,c,rho):
        cs = []
        for ind, ci in enumerate(c):
            cs.append(self.beta_i[ind] / self.L * n - self.lam[ind] * ci)
        return cs
    
    def fci(self, group, n, c, rho):
        return self.beta_i[group] / self.L * n - self.lam[group] * c

        
def main():
    kilo = krusty()
    kilo.setInputs()
    kilo.Heun()

if __name__ == "__main__":
    main()