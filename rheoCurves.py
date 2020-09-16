import numpy as np
# Load the Pandas libraries with alias 'pd' 
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.font_manager

matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


# Time dependencies
def tauY(t,ts,tau0):
    return tau0*np.exp(t/ts)

def eta0t(t,ts,eta0):
    return eta0*np.exp(t/ts)

# Herschel-Bulkley Model
def HB_tau(gammaDot,t,nPow,K,tauY0,ts):
    eps = 1e-20
    # tauY_t = tauY0              #Constant tau0
    tauY_t = tauY(t,ts,tauY0)   #Time-dependend tau0
    return (tauY_t + K*(np.power(gammaDot,nPow)))
def HB_eta(gammaDot,t,nPow,K,tauY0,ts):
    eps = 1e-20
    # tauY_t = tauY0              #Constant tau0
    tauY_t = tauY(t,ts,tauY0)   #Time-dependend tau0
    return HB_tau(gammaDot,t,nPow,K,tauY0,ts)/(gammaDot+eps)

# Herschel-Bulkley Modified Model
def HBMod_tau(gammaDot,t,nPow,K,tauY0,ts,eta0):
    eps = 1e-20
    # tauY_t = tauY0              #Constant tau0
    tauY_t = tauY(t,ts,tauY0)   #Time-dependend tau0
    return (tauY_t + K*(np.power(gammaDot,nPow)) + eta0*gammaDot)
def HBMod_eta(gammaDot,t,nPow,K,tauY0,ts,eta0):
    eps = 1e-20
    # tauY_t = tauY0              #Constant tau0
    tauY_t = tauY(t,ts,tauY0)   #Time-dependend tau0
    return HBMod_tau(gammaDot,t,nPow,K,tauY0,ts,eta0)/(gammaDot+eps)

# SMD Model
def SMDMod_tau(gammaDot,t,nPow,K,tauY0,ts,eta0,etaInf):
    eps = 1e-20
    # tauY_t = tauY0              #Constant tau0
    tauY_t = tauY(t,ts,tauY0)   #Time-dependend tau0
    eta0_t = eta0               #Constant eta0
    # eta0_t = eta0t(t,ts,eta0) #Time-dependend eta0
    return (1 - np.exp(-eta0_t*(gammaDot)/tauY_t))* (tauY_t + K*(np.power(gammaDot,nPow)))

def SMDMod_eta(gammaDot,t,nPow,K,tauY0,ts,eta0,etaInf):
    eps = 1e-10
    # tauY_t = tauY0              #Constant tau0
    tauY_t = tauY(t,ts,tauY0)   #Time-dependend tau0
    eta0_t = eta0               #Constant eta0
    # eta0_t = eta0t(t,ts,eta0) #Time-dependend eta0
    return (1 - np.exp((-eta0_t*(gammaDot))/tauY_t))*(tauY_t/(gammaDot+eps) + K*(pow(gammaDot+eps,nPow-1)) + etaInf)

ExpCurvePWS = np.array([[0.209,10.899],
                        [0.332,11.136],
                        [0.578,11.297],
                        [1.006, 11.626],
                        [1.73, 11.964],
                        [3.01, 12.58],
                        [5.236, 13.133],
                        [9.11, 14.11],
                        [15.849, 15.269],
                        [27.573, 17.498],
                        [47.971, 21.39],
                        [83.459, 25.962]])

# Saving Inputs
detail = '_LabSample'
figFormat = '.png'

# Parameters
# PWS Sample
# eta0 = 130
# etaInf = 0.2
# nPow = 0.98
# K = 0.2
# tauY0 = 12
ts = 8000
# Lab Sample
eta0 = 5000
etaInf = 0.292
nPow = 0.54
K = 1.43
tauY0 = 19.79
ts = 8000

paramsSMD = [nPow,K,tauY0,ts,eta0,etaInf]
paramsHB = [nPow,K,tauY0,ts]
paramsHBMod = [nPow,K,tauY0,ts,eta0]

## Calculations
gammaDot = np.logspace(-2,2,num=1000)
t = np.logspace(0,4,num=1000)

# HB
tauHB = HB_tau(gammaDot,1e2,*paramsHB)
etaHB = HB_eta(gammaDot,1e2,*paramsHB)
# HBMod
tauHBMod = HBMod_tau(gammaDot,1e2,*paramsHBMod)
etaHBMod = HBMod_eta(gammaDot,1e2,*paramsHBMod)
# SMD
tauSMD = SMDMod_tau(gammaDot,1e2,*paramsSMD)
etaSMD = SMDMod_eta(0.1,t,*paramsSMD)

# tau_Y = tauY(t,ts,tau0)

## Plotting
# Figures
fig1,ax1 = plt.subplots()
fig2,ax2 = plt.subplots()

# Plots
ax1.loglog(t,etaHB,label='HBMod')
ax1.loglog(t,etaSMD,label='SMDMod')
ax2.loglog(gammaDot,tauHB,label='HBMod')
ax2.loglog(gammaDot,tauSMD,label='SMDMod')

# Beauty
ax1.set_xlabel('$t(s)$')
ax1.set_ylabel('$\eta(Pa.s)$')
ax1.legend()
ax2.set_xlabel('$\dot{\gamma}(s^{-1})$')
ax2.set_ylabel(r'$\tau (Pa)$')
ax2.legend()

# plt.show()
# Saving
fig1.savefig('./Images/rheoFitting_Eta'+detail+figFormat,dpi=400)
fig2.savefig('./Images/rheoFitting_Tau'+detail+figFormat,dpi=400)