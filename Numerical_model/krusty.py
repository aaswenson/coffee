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
    def __init__(self):
        self.n0 = 1                     # initial neutron population
        enrich = 0.93155                # U-235 enrichment
        Moly_wo = 0.075                 # Moly weight fraction [wt%]
        rho_cost = 0.60                 # Initial step reactivity cost [$]
        self.reactor()                  # set Physical parameters of system
        self.nuclear_data(rho_cost)     # set 6-group data
        self.N235 = self.num_dens(enrich, Moly_wo)     # set U235 number density
    
    def nuclear_data(self,rho=0.6):
        # Fission parameters
        self.L = float(2.4e-5)                  # mean neutron lifetime
        self.nu_235 = float(2.57)               # Neutrons per fission, U-235 Fast Fission
        self.E_fission = float(185) * unit['MeV_J']
        self.sigma_F = 2 * unit['b_cm']         # Fission cross-section [cm^2]
        self.n_W = self.E_fission/self.nu_235   # conversion factor from neutron to watts
        # Decay constant, lambda [sec^-1] (D. Hetrick, Table 1-2, p.11)
        self.lam = [0.0127, 0.0317, 0.115, 0.311, 1.40, 3.87]
        # Delayed Neutron Yield [n/fission]
        self.beta_nu = [0.00063, 0.00351, 0.00310, 0.00672, 0.00211, 0.00043]
        # Delayed Neutron Fractions
        self.beta_i = np.divide(self.beta_nu, self.nu_235)
        self.beta = sum(self.beta_i)            # Total Beta Fraction
        self.groups = len(self.lam)             # Number of delayed groups
        # Initial Precursor Vectors
        self.c0 = [[beta*self.n0/(self.L*lam)] for beta, lam in zip(self.beta_i, self.lam)]
        self.rho0 = rho*self.beta
        # # K-factor
        self.Kf = self.n_W/self.C_th            # k-factor

        # # Neutron speed
        # E_n = 0.5 * unit['MeV_J']   # averege neutron  energy ~ 500 keV
        # v_n = np.sqrt(2*E_n/unit['mass_n'])*100 # neutron speed [cm/s]

    def num_dens(self, enr=0.93155, w_Mo = 0.075):
        """ return number density of U-235"""
        M_U = (enr/amass['U235'] + (1-enr)/amass['U238'])**-1 # adjusted M Uranium
        at_Mo = (w_Mo/amass['Mo'])/(w_Mo/amass['Mo'] + (1-w_Mo)/M_U)  # atomic fraction Mo
        at_U = 1-at_Mo          # atomic fraction U
        M_UMo = M_U*at_U + amass['Mo']*at_Mo   # adjusted M for U-Mo mixture
        rho = 17.6              # nominal density 
        return enr*unit['Av']/amass['U235'] * rho * M_U/M_UMo
    
    def reactor(self):
        self.volume = float(1940)           # Fuel volume [cc]
        N_ch = 8                            # Number of coolant channels
        D_ch = 0.0107                         # ID of heat pipe
        A_ch = np.pi*D_ch**2/4              # cross-sectional area of heat pipe
        self.A_chs = N_ch * A_ch            # Total cross-sect. area of heat pipes
        self.Tf0 = float(20)                # Initial Fuel Temperature [C]
        dens_f = self.dens(self.Tf0)        # [g/cc]
        cp_f = self.cp(self.Tf0)            # [J/g-C]
        # # Thermal Capacity
        self.C_th = dens_f*cp_f*self.volume # [g/cc]*[J/g-C]*[cc] = [J/C]
        r_fi = 0.02                         # inner radius, fuel [m]
        r_fo = 0.055                        # outter radius, fuel [m]
        hgt = 0.25                          # Reactor height [m]
        self.R_f0 = np.log(r_fo/r_fi)/(2*np.pi*hgt) # resitance term

    def RungeKutta4(self):        
        Ttime = float(20*unit['sec_min'])                  # Simulation Time
        dt = 0.01                           # Time step size
        tsec = np.arange(0, Ttime+dt, dt)   # time vector
        rho = [self.rho0]                   # value vectors
        rho_j0 = rho[0]
        T_0 = 20
        rhof = [0]
        Tf = [self.Tf0]
        Tc = [self.Tf0]
        n = [self.n0]
        c = self.c0
        order = 4                           # order of approach
        for ind, t in enumerate(tsec[1:]):
            print(t,Tf[-1],rho[-1])
            # initial values
            n_j0 = n[-1]
            c_j0 = [x[-1] for x in c]
            T_j0 = Tf[-1]
            Tcj0 = Tc[-1]
            if T_j0 >= 350:
                rho_j0 = 0
                T_0 = 400
            rho_j = self.rho_feedback(rho_j0,T_j0,T_0)
            # print(t,n_j0,T_j0,rho_j0)
            # if T_j0 >= 400:
                # rho0 = 0
            # rho_j = self.frho(rho0,T_j0,n_j0)

            # first derivative
            dna = self.fn(n_j0, c_j0, rho_j)
            dca = [f for f in self.fc(n_j0, c_j0, rho_j)]
            dTa = self.fTemp(n_j0, T_j0)
            #
            n_j1 = n_j0 + dna*dt/2
            c_j1 = [c + dc*dt/2 for c, dc, in zip(c_j0, dca)]
            T_j1 = T_j0 + dTa*dt/2
            #
            dnb = self.fn(n_j1, c_j1, rho_j0)
            dcb = [f for f in self.fc(n_j1, c_j1, rho_j0)]
            dTb = self.fTemp(n_j1, T_j1)
            #
            n_j2 = n_j0 + dnb*dt/2
            c_j2 = [c + dc*dt/2 for c, dc, in zip(c_j0, dcb)]
            T_j2 = T_j0 + dTb*dt/2
            #
            dnc = self.fn(n_j2, c_j2, rho_j0)
            dcc = [f for f in self.fc(n_j2, c_j2, rho_j0)]
            dTc = self.fTemp(n_j2, T_j2)
            #
            n_j3 = n_j0 + dnc*dt
            c_j3 = [c + dc*dt for c, dc, in zip(c_j0, dcc)]
            T_j3 = T_j0 + dTc*dt/2
            #
            dnd = self.fn(n_j3, c_j3, rho_j0)
            dcd = [f for f in self.fc(n_j3, c_j3, rho_j0)]
            dTd = self.fTemp(n_j3, T_j3)
            #
            n_j4 = n_j0 + (dna + 2*dnb + 2*dnc + dnd)*dt/6
            c_j4 = [cj + (a + 2*b + 2*c + d)*dt/6 for cj, a, b,
                    c, d in zip(c_j0, dca, dcb, dcc, dcd)]
            T_j4 = T_j0 + (dTa + 2*dTb + 2*dTc + dTd)*dt/6
            n.append(n_j4)
            Tf.append(T_j4)
            for i in range(self.groups):
                c[i].append(c_j4[i])    
            rho.append(rho_j)

        self.plot_results(tsec,n,rho,Tf,c)


    def Heun(self):
        rho_cost = float(1)              # Reactivity [$]
        rho0 = rho_cost*self.beta           # Reactivity [delta(k)/k]
        alpha = 0.15                        # Fuel reactivity temperature coefficient
        K = 0.017                            # [C/kw-sec]
        Tf0 = float(20)                     # Initial Fuel Temperature [C]
        Ttime = float(10*60)               # total time [3 hours]
        Nstep = float(Ttime*1000)                  # total steps
        dt = Ttime/Nstep                    # time step size
        tsec = np.arange(0, Ttime+dt, dt)
        rho = [rho0]
        Tf = [Tf0]
        n = []
        c = []
        n = [self.n0]
        c = self.c
        for ind, tm in enumerate(tsec[1:]):
            # precursors from last time step
            clast = [x[-1] for x in c]
            n_pred = n[-1] + dt * \
                self.fn(n[-1], clast, rho[-1])    # predictor step
            c_pred = ([cl + dt*f for cl, f in zip(clast,
                                                  self.fc(n[-1], clast, rho[-1]))])
            rho = [rho0 - alpha*(Tf[-1]-Tf0)]

            n_corr = n[-1] + dt/2*(self.fn(n[-1], clast,
                                           rho[-1])+self.fn(n_pred, c_pred, rho[-1]))
            n.append(n_corr)
            for i in range(self.groups):
                c[i].append(clast[i]+dt/2*(self.fci(i, n[-1], clast[i],
                                                    rho[-1]) + self.fci(i, n_pred, c_pred[i], rho[-1])))

    def fn(self, n, c, rho):
        """ Neutron derivative function """
        return (rho-self.beta)/self.L*n + np.dot(c, self.lam)

    def fc(self, n, c, rho):
        """ 6-group precursor derivative function """
        cs = []
        for ind, ci in enumerate(c):
            cs.append(self.beta_i[ind] / self.L * n - self.lam[ind] * ci)
        return cs

    def fci(self, group, n, c, rho):
        """ Single group (of 6) precursor derivative function """
        return self.beta_i[group] / self.L * n - self.lam[group] * c

    # def fcool(self,)

    def fTemp(self, n, T):
        """ Fuel Temperautre function """
        # return (self.H0*n/(self.dens(T)*self.cp(T) * self.volume))
        P_fuel = self.Kf * n                # Reactor power
        deltaT = T/20                         # Temp. diff. from fuel to coolant(bulk)
        h_bar = 20000                         # [W/m^2-K]
        R_f = self.R_f0/self.kcond(T)
        R_conv = 1/(h_bar*self.A_chs)
        R_total = R_f + R_conv
        if T <= 100:
            Q_out = 0
        else:
            Q_out = deltaT/R_total
        return (P_fuel - Q_out)/self.C_th

    def frho(self,rho0,T,n):
        """ Reactivity with Temperature Feedback """
        # return float(rho0 + self.RTC(T)*(T-T0))
        h_bar = 20000
        rho_new =  float(rho0 + self.RTC(T)*(self.Kf*n - 2*(h_bar*self.A_chs)/self.C_th) )
        # rho_new = rho0  + self.RTC(T)*(T-20)
        print(T,'\t',rho_new)
        return rho_new
    
    def rho_feedback(self,rho0,T,T0):
        
        return rho0 + self.RTC(T)*(T-T0)

    def RTC(self, T):
        """ Fuel Temperautre Reactivity Coeficient [K^-1] """
        alpha = self.beta*(-7.3e-9*(T+unit['C_K'])**2 -7.58e-5*(T+unit['C_K']) -0.113)/100
        return alpha

    def cp(self, T):
        """ Fuel Specific Heat [J/g-C] """
        # Eq.1 (INL/EXT-10-19373 Thermophysical Properties of U-10Mo Alloy)
        # Range: 100 < T < 1000 [C]
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
        plt.figure(1)
        plt.plot(time,n)
        plt.title('Neutron Density')
        plt.xlabel('Time [sec]')
        plt.ylabel('n(t)')
        plt.yscale('log')
        # Reactivity
        plt.figure(2)
        plt.plot(time,rho)
        plt.title('Reactivity')
        plt.xlabel('Time [sec]')
        plt.ylabel(r'$\rho(t)$')
        plt.yscale('linear')
        # Fuel Temperature
        plt.figure(3)
        plt.plot(time,Tf, label='Fuel Temp')
        plt.title('Fuel Temperature')
        plt.xlabel('Time [sec]')
        plt.ylabel(r'T_f(t)')
        plt.yscale('linear')
        # Reactor Power
        plt.figure(4)
        power = np.multiply(n,self.n_W)
        plt.plot(time,power,label = 'watts')
        plt.xlabel('Time [sec]')
        plt.ylabel('P(t)')
        plt.yscale('log')
        plt.show()


def main():
    start = time.time()
    kilo = krusty()
    # kilo.Heun()
    kilo.RungeKutta4()
    end = time.time()
    print("Calculation time: %0.f" % (end-start))


if __name__ == "__main__":
    main()
