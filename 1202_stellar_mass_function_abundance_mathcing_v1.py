import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import gamma as ga
import math
import pickle
import numpy as np
from scipy import integrate
import time
import matplotlib.pyplot as plt
from termcolor import colored
import h5py
import math
import os
from helpers.SimulationAnalysis import SimulationAnalysis, readHlist, iterTrees, getMainBranch
import pickle
import emcee
import scipy.optimize as op
omega_m = 0.272
omega_gamma = 0.728

pkl_file = open('a_index_us.pkl', 'rb')
a_index_us = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('a_index_Behroozi.pkl', 'rb')
a_index_Behroozi = pickle.load(pkl_file)
pkl_file.close()

# weight factor in a
pkl_file = open('a_weight_factor.pkl', 'rb')
a_weight_factor = pickle.load(pkl_file)
pkl_file.close()

# read halo mass function:

pkl_file = open("HMF_11_14.pkl", 'rb')
HMF = pickle.load(pkl_file)
pkl_file.close()


def E(z):
    return (omega_m * (1 + z) ** 3 + omega_gamma) ** 0.5


def a_to_time_Hogg(a):
    z = 1 / a - 1

    result = integrate.quad(lambda x: ((1 + x) * E(x)) ** (-1), z, np.inf)

    return result[0] / 8.75428526796e-11


a_to_time_Hogg = np.vectorize(a_to_time_Hogg)

# Quenching fraction:
tinker = np.loadtxt("tinker_fqcen_SDSS_M9.7.dat")

ms_tinker = tinker[:, 0]
fraction_tinker = tinker[:, 1]
error_tinker = tinker[:, 2]



def log10(x):
    if x > 0:
        return math.log10(x)
    else:
        return -np.inf


def exp(x):
    try:
        return math.exp(x)
    except:
        return np.inf


exp = np.vectorize(exp)
log10 = np.vectorize(log10)




# line 0 logMgal
# line 1 log SMF
# line 2 error

# plot:


# Let's read and plot Behroozi
plot_path = "/Users/caojunzhi/Downloads/upload_201712_Jeremy/"

#

Sersic = np.genfromtxt("MsF_Ser.dat")


smf_p = np.poly1d(np.polyfit(Sersic[:,0], Sersic[:,1], 10))




path = "HMF_8_to_16/"+"all_plots/"+"mVector_PLANCK-SMT .txt"

fusion = np.loadtxt(path)

#print(fusion[:,0])
#print(fusion[:,7])

"""


plt.plot(log10(fusion[:,0]),log10(fusion[:,7]),"ko")
plt.plot(Sersic[:,0],Sersic[:,1],"ro")
plt.plot()

"""


"""

# [1] m:            [M_sun/h] 
# [2] sigma 
# [3] ln(1/sigma) 
# [4] n_eff 
# [5] f(sigma) 
# [6] dn/dm:        [h^4/(Mpc^3*M_sun)] 
# [7] dn/dlnm:      [h^3/Mpc^3] 
# [8] dn/dlog10m:   [h^3/Mpc^3] 
# [9] n(>m):        [h^3/Mpc^3] 
# [11] rho(>m):     [M_sun*h^2/Mpc^3] 
# [11] rho(<m):     [M_sun*h^2/Mpc^3] 
# [12] Lbox(N=1):   [Mpc/h]
"""



"""



plt.plot(Sersic[:,0],Sersic[:,1],'ko')
plt.plot(Sersic[:,0],smf_p(Sersic[:,0]),"r")
plt.show()


"""





hmf_p = np.poly1d(np.polyfit(log10(fusion[:,0]), log10(fusion[:,7]), 10))


"""



plt.plot(log10(fusion[:,0]),log10(fusion[:,7]),'ko')
plt.plot(log10(fusion[:,0]),hmf_p(log10(fusion[:,0])),"r")
plt.show()



"""


## construct function for HMF and SMF

# use Intergral phi * dMgal = phi * mgal dlogMgal ??

def calculate_abundance_hmf(threshold):

    return integrate.quad(lambda x: 10**hmf_p(x), threshold, log10(fusion[-1,0]))[0]


def calculate_abundance_smf(threshold):
    return integrate.quad(lambda x: 10**smf_p(x), threshold, Sersic[-1,0])[0]

# print(Sersic[:,0])
# calculate smf at each stellar mass bins from 9.05 to 12.35 with size=0.1


smf_values = []

for i in range(0, 34):
    smf_values.append(calculate_abundance_smf(threshold=9.05+i*0.1))

smf_values = np.array(smf_values)

# print(smf_values)


# calculate hmf integral for all halos:


path_catalog = "/Users/caojunzhi/Desktop/hlist_1.00231.h5"

f = h5py.File(path_catalog, 'r')

mvir_array = f["data"][:,0]

mvir_array_log = log10(mvir_array)


f.close()

"""

plt.hist(mvir_array_log,bins=50,facecolor='red')

plt.show()



"""


print("calculating hmf values")
print(smf_values)

mask_smf = smf_values<0

smf_values[mask_smf] = 0

hmf_values = []

for j in range(0,len(mvir_array_log)):

    print("Doing %.2f percent"%(j/len(mvir_array_log)*100))

    hmf_values.append(calculate_abundance_hmf(threshold=mvir_array_log[j]))

hmf_values = np.array(hmf_values)

# save hmf_values

output = open("hmf_values_z0.pkl", 'wb')
pickle.dump(hmf_values, output)
output.close()


plt.plot(hmf_values,"ro")

plt.show()




