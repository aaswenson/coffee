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
        self.L = float(2.4e-5)
        self.n0 = float(1)
        self.n = [self.n0]

        # D. Hetrick, Table 1-2, p.11
        # Decay constant, lambda [sec^-1]
        self.lam = [0.0127, 0.0317, 0.115, 0.311, 1.40, 3.87]
        # Neutrons per fission, U-235 Fast Fission, nu
        self.nu_235 = float(2.57)
        # Delayed Neutron Yield [n/fission]
        self.beta_nu = [0.00063, 0.00351, 0.00310, 0.00672, 0.00211, 0.00043]
        # Delayed Neutron Fractions
        self.beta_i = np.divide(self.beta_nu, self.nu_235)
        # Total Beta Fraction
        self.beta = sum(self.beta_i)
        # Number of delayed groups
        self.groups = len(self.lam)
        # Initial Precursor Vectors
        self.c = [[beta*self.n0/(self.L*lam)]
                  for beta, lam in zip(self.beta_i, self.lam)]

    def number_density(self):
        """ return number density of U-235"""
        enr = 0.93155           # U-235 enrichment
        w_Mo = 0.075            # weight fraction of Mo in U-Mo
        M235 = 235.04           # M U-235
        M238 = 238.05           # M U-238
        M_Mo = 95.94            # M nat-Mo
        M_U = (enr/M235 + (1-enr)/M238)**-1 # adjusted M Uranium
        at_Mo = (w_Mo/M_Mo)/(w_Mo/M_Mo + (1-w_Mo)/M_U)  # atomic fraction Mo
        at_U = 1-at_Mo          # atomic fraction U
        M_UMo = M_U*at_U + M_Mo*at_Mo
        rho = 17.6; rho_U = 16.3
        return enr*self.Av/M235 * rho * M_U/M_UMo

    def setInputs(self):
        self.MeV_J = float(1.6021892e-13)       # Joule/MeV conversion
        self.boltz = float(1.380662e-23)        # Boltzmann constant,k [J/K]
        self.mass_n = float(1.674929e-27)       # mass of neutron [kg]
        self.Av = float(6.0225e23)
        self.C_K = float(273.15)
        self.N_235 = self.number_density()      # U235 number density [at/cc]
        Power_rated = float(5e3)                # Rated thermal power
        Tf_ave = float(800)                     # deg C 
        Tc_ave = float(750)                     # deg C
        M_235 = (27.5e3)                        # grams U-235
        phi_ave = float(9.3e11)                 # Core Average Neutron Flux
        nom_Wcc = float(1.6)                # Nominal Power density
        self.volume = float(1940)           # Fuel volume [cc]
        self.Efis = float(190)*self.MeV_J   # Recoverable energy from fission [MeV] --> [J]
        self.E_neutron = float(0.5)*self.MeV_J         # Average neutron energy is 500 keV
        # Average neutron speed [m/s]       
        self.v_n = np.sqrt(2*self.E_neutron/self.mass_n)*100    # neutron speed [cm/s]
        self.sigma_F = float(2e-24)                             # fast fission xs [cm^2]
        self.H0 = self.v_n * self.Efis * self.sigma_F * self.N_235 /self.volume   # [cm/s * J/fis * cm^2 * at/cm^3]
        self.H1 = 17.6 * 36 * self.volume
        self.k_factor = self.H1 * self.H0

    def RungeKutta4(self):
        rho_cost = float(0.6)               # Initial step reactivity cost [$]
        rho0 = rho_cost * self.beta         # Reactivity [dk/k]
        Tf0 = float(20)                     # Initial Fuel Temperature [C]
        Ttime = float(60*10)                # Simulation Time
        dt = 0.01                           # Time step size
        tsec = np.arange(0, Ttime+dt, dt)   # time vector
        rho = [rho0]                        # value vectors
        Tf = [Tf0]
        n = []
        c = []
        n = [self.n0]
        c = self.c
        order = 4                           # order of approach
        for ind, t in enumerate(tsec[1:]):
            # initial values
            n_j0 = n[-1]
            c_j0 = [x[-1] for x in c]
            T_j0 = Tf[-1]
            rho_j0 = rho[-1]
            rho_j = self.frho(rho0,T_j0,n_j0)
            if rho_j <= float(0):
                rho_j = 0
            # first derivative
            dna = self.fn(n_j0, c_j0, rho_j0)
            dca = [f for f in self.fc(n_j0, c_j0, rho_j0)]
            dTa = self.fTemp(n_j0, T_j0)
            #
            n_j1 = n_j0 + dna*dt/2
            c_j1 = [c + dc*dt/2 for c, dc, in zip(c_j0, dca)]
            T_j1 = T_j0 + dTa*dt/2
            #
            dnb = self.fn(n_j1, c_j1, rho_j)
            dcb = [f for f in self.fc(n_j1, c_j1, rho_j)]
            dTb = self.fTemp(n_j1, T_j1)
            #
            n_j2 = n_j0 + dnb*dt/2
            c_j2 = [c + dc*dt/2 for c, dc, in zip(c_j0, dcb)]
            T_j2 = T_j0 + dTb*dt/2
            #
            dnc = self.fn(n_j2, c_j2, rho_j)
            dcc = [f for f in self.fc(n_j2, c_j2, rho_j)]
            dTc = self.fTemp(n_j2, T_j2)
            #
            n_j3 = n_j0 + dnc*dt
            c_j3 = [c + dc*dt for c, dc, in zip(c_j0, dcc)]
            T_j3 = T_j0 + dTc*dt/2
            #
            dnd = self.fn(n_j3, c_j3, rho_j)
            dcd = [f for f in self.fc(n_j3, c_j3, rho_j)]
            dTd = self.fTemp(n_j3, T_j3)
            #
            n_j4 = n_j0 + (dna + 2*dnb + 2*dnc + dnd)*dt/6
            c_j4 = [cj + (a + 2*b + 2*c + d)*dt/6 for cj, a, b,
                    c, d in zip(c_j0, dca, dcb, dcc, dcd)]
            T_j4 = T_j0 + (dTa + 2*dTb + 2*dTc + dTd)*dt/6
            # print(t,n_j4,rho[-1])
            n.append(n_j4)
            Tf.append(T_j4)
            for i in range(self.groups):
                c[i].append(c_j4[i])    
            rho.append(rho_j)

        tmin = np.divide(tsec,60)
        plt.figure(1)
        plt.plot(tmin,n)
        plt.title('Neutron Density')
        plt.xlabel('Time [sec]')
        plt.ylabel('n(t)')
        plt.yscale('log')
        plt.figure(2)
        plt.plot(tsec,rho)
        plt.title('Reactivity')
        plt.xlabel('Time [sec]')
        plt.ylabel(r'$\rho(t)$')
        plt.yscale('log')
        plt.figure(3)
        plt.plot(tsec,Tf)
        plt.title('Fuel Temperature')
        plt.xlabel('Time [sec]')
        plt.ylabel(r'T_f(t)')
        plt.yscale('linear')
        # plt.yscale('log')
        plt.show()

    def Heun(self):
        rho_cost = float(0.6)              # Reactivity [$]
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
        print('done')
        tmin = np.divide(tsec, float(60))
        # plt.plot(tmin,n,label=rho_cost)
        # plt.yscale('log')
        # plt.ylim(0,1e8)
        n = 0
        # plt.legend()
        # plt.show()

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

    def fTemp(self, n, T):
        return (self.k_factor*n/(self.volume*self.dens(T)*self.cp(T)))

    def frho(self,rho0,T,n):
    
        # return float(rho0 + self.RTC(T)*(T-T0))
        return float(rho0 + self.beta*self.RTC(T)*self.k_factor*n)

    def RTC(self, T):
        # """ Third-Order Fit to Fuel Temperature Reactivity Coefficient (INSTANTANEOUS)"""
        alpha =  float(3.87e-11*(T+self.C_K) ** 3 - 1.06e-7*(T+self.C_K) ** 2 - 1.43e-7*(T+self.C_K) - 0.13)/100
        # return alpha
        """ Third-Order Fit to Fuel Temperature Reactivity Coefficient (INTEGRAL)"""
        # alpha =  float(1.601e-13 *(T+self.C_K) ** 3 - 4.965e-10*(T+self.C_K) ** 2 + 5.147e-8*(T+self.C_K) - 1.373e-3)
        return alpha

    def cp(self, T):
        """ Fuel Specific Heat [J/g-C] """
        # Eq.1 (INL/EXT-10-19373 Thermophysical Properties of U-10Mo Alloy)
        # 100 < T < 1000 [C]
        return float(0.137 + 5.12e-5*T + 1.99e-8*T**2)

    def dens(self, T):
        """ Fuel Density [g/cc] """
        # Eq.4 (INL/EXT-10-19373 Thermophysical Properties of U-10Mo Alloy)
        # 20 < T < 700 [C]
        return float(17.15 - 8.63e-4*(T+20))

    def kcond(self, T):
        """ Fuel Thermal conductivity [W/m-C] """
        # Eq.4 (INL/EXT-10-19373 Thermophysical Properties of U-10Mo Alloy)
        # 20 < T < 800 [C]
        return float(10.2 + 3.51e-2*T)
        # Haynes-230    return (float(0.02*T + 8.4315))


def main():
    start = time.time()
    kilo = krusty()
    kilo.setInputs()
    # kilo.Heun()
    kilo.RungeKutta4()
    end = time.time()
    print("Calculation time: %0.f" % (end-start))


if __name__ == "__main__":
    main()
