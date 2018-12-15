import numpy as np

unit = {'Av'    : float(6.0225e23),
         'MeV_J' : float(1.6021892e-13),
         'boltz' : float(1.380662e-23),
         'mass_n': float(1.674929e-27),
         'C_K'   : float(273.15),
         'b_cm'  : float(1e-24),
         'sec_min': float(60)
         }
# Avogadro's number  [mol^-1]
# J/MeV conversion   [J/MeV]
# Boltzmann constant [J/K]
# Neutron rest mass  [kg]
# Celcius to Kelvin
# barn to cm^2

# # Atomic Masses
amass = {'U235': float(235.0439299),
         'U238': float(238.05078826),
         'Pu239' : float(239.0521634),
         'Mo'  : float(95.95)
        }

dens = {'U235' : float(19.1),
        'Pu239' : float(19.86)
       }

ins1 = lambda T: 0.15
ins2 = lambda T: Reactivity.feedback(beta, T)

class Reactivity(self):
    """
    """
    def __init__(self, times, insertion):
        """
        """
        self.times = times
        self.ins = insertion
    
    @staticmethod
    def feedback(beta, T)
        """Calculate temperature coefficient of reactivity
        """
        # Reactivity temperature coefficient [dk/k/K]
        RTC = beta*(-7.3e-11*(T)**2 -7.58e-7*(T) - 1.13e-3)


    def set_function(self, t, T):
        """
        """

        funcdx = np.searchsorted(times, t, 'left') - 1
        rhofunc = self.ins[funcdx]

        rho_ins = rhofunc(T)
        rho_feed = 


    def rho(self, added_rho):
        """Compute rho with added_rho and feedback
        """

class FuelMat:
    # Decay constant, lambda [sec^-1] (D. Hetrick, Table 1-2, p.11)
    lams = {'U235'  : np.array([0.0127, 0.0317, 0.115, 0.311, 1.40, 3.87]),
            'Pu239' : np.array([0.0129, 0.0311, 0.134, 0.331, 1.26, 3.21])
           }
    # Delayed Neutron Yield [n/fission]
    beta_nu = {'U235'  : np.array([0.00063, 0.00351, 0.00310, 0.00672, 0.00211, 0.00043]),
               'Pu239' : np.array([0.00024, 0.00176, 0.00136, 0.00207, 0.00065, 0.00022])
              }
    
    nuc_data = {'U235'  : {'efiss' : 185,   'nu' : 2.57},
                'Pu239' : {'efiss' : 211.5, 'nu' : 2.88}
               }

    def __init__(self, Pufrac, enrich=0.93155):

        self.pu_mfrac = Pufrac
        self.enrich = enrich

        self.mix_mats()
        self.nuc_data_make()

    def mix_mats(self, enr=0.93155):
        """ return number density of U-235"""
        
        mfrac_235 = (1 - self.pu_mfrac)*enr
        
        N235 = mfrac_235 * dens['U235'] * unit['Av'] / amass['U235']
        N239 = self.pu_mfrac * dens['Pu239'] * unit['Av'] / amass['U235']
        
        Ntot = N235 + N239
        self.afrac = {'U235' : N235/ Ntot, 'Pu239' : N239 / Ntot}
    
    def nuc_data_make(self):
        # Fission parameters
        self.L = 2.4e-5
        self.E_fission = 0
        self.nu = 0
        self.beta_i = np.zeros(6)
        self.lam = np.zeros(6)
        for fuel in self.nuc_data:
            self.nu += self.nuc_data[fuel]['nu']*self.afrac[fuel]
            self.E_fission += self.nuc_data[fuel]['efiss'] * self.afrac[fuel] *\
                              unit['MeV_J']
            self.beta_i += self.beta_nu[fuel] * self.afrac[fuel] /\
                                                self.nuc_data[fuel]['nu']
            self.lam += self.lams[fuel] * self.afrac[fuel]
        
        self.n_W = self.E_fission/(self.nu)  # conversion factor from neutron to watts
        # Delayed Neutron Fractions
        self.beta = sum(self.beta_i)            # Total Beta Fraction
        self.groups = len(self.lam)             # Number of delayed groups
        # Initial Precursor Vectors
        self.c0 = np.divide(self.beta_i, np.multiply(self.lam, self.L))
