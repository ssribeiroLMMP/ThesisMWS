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
    tauY_t = tauY0              #Constant tau0
    # tauY_t = tauY(t,ts,tauY0)   #Time-dependend tau0
    return (tauY_t + K*(np.power(gammaDot,nPow)))+0*t
def HB_eta(gammaDot,t,nPow,K,tauY0,ts):
    eps = 1e-20
    tauY_t = tauY0              #Constant tau0
    # tauY_t = tauY(t,ts,tauY0)   #Time-dependend tau0
    return HB_tau(gammaDot,t,nPow,K,tauY0,ts)/(gammaDot+eps)

# # Herschel-Bulkley Modified Model
# def HBMod_tau(gammaDot,t,nPow,K,tauY0,ts,eta0):
#     eps = 1e-20
#     # tauY_t = tauY0              #Constant tau0
#     tauY_t = tauY(t,ts,tauY0)   #Time-dependend tau0
#     return (tauY_t + K*(np.power(gammaDot,nPow)) + eta0*gammaDot)
# def HBMod_eta(gammaDot,t,nPow,K,tauY0,ts,eta0):
#     eps = 1e-20
#     # tauY_t = tauY0              #Constant tau0
#     tauY_t = tauY(t,ts,tauY0)   #Time-dependend tau0
#     return HBMod_tau(gammaDot,t,nPow,K,tauY0,ts,eta0)/(gammaDot+eps)

# SMD Model
def SMDMod_tau(gammaDot,t,nPow,K,tauY0,ts,eta0,etaInf):
    eps = 1e-20
    # tauY_t = tauY0              #Constant tau0
    tauY_t = tauY(t,ts,tauY0)   #Time-dependend tau0
    # eta0_t = eta0               #Constant eta0
    eta0_t = eta0t(t,ts,eta0) #Time-dependend eta0
    return (1 - np.exp(-eta0_t*(gammaDot)/tauY_t))* (tauY_t + K*(np.power(gammaDot,nPow)))

def SMDMod_eta(gammaDot,t,nPow,K,tauY0,ts,eta0,etaInf):
    eps = 1e-10
    # tauY_t = tauY0              #Constant tau0
    tauY_t = tauY(t,ts,tauY0)   #Time-dependend tau0
    # eta0_t = eta0               #Constant eta0
    eta0_t = eta0t(t,ts,eta0) #Time-dependend eta0
    return (1 - np.exp((-eta0_t*(gammaDot))/tauY_t))*(tauY_t/(gammaDot+eps) + K*(pow(gammaDot+eps,nPow-1)) + etaInf)

ExpCurveLAB = np.array([[483.548, 136.529],
                        [285.713, 94.864],
                        [166.717, 69.308],
                        [98.508, 53.245],
                        [9.826, 22.673],
                        [0.98, 22.11],
                        [0.089, 35.15],
                        [0.01, 54]])

ExpCurvePWS = np.array([[0.012,17.899],
                        [0.025,12.780],
                        [0.049,10.980],
                        [0.1,10.901],
                        # [0.130,10.900],
                        [0.209,10.899],
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
detail = 'Comparison'
figFormat = '.PNG'

# Parameters
# PWS Sample
eta0 = 700
etaInf = 0.3
nPow = 0.79
K = 0.46
tauY0 = 11
ts = 8000
paramsPWS_SMD = [nPow,K,tauY0,ts,eta0,etaInf]
paramsPWS_HB = [nPow,K,tauY0,ts]
paramsPWS_HBMod = [nPow,K,tauY0,ts,eta0]

# Lab Sample
eta0 = 1300
etaInf = 0.292
nPow = 0.54
K = 1.43
tauY0 = 19.79
ts = 8000
paramsLAB_SMD = [nPow,K,tauY0,ts,eta0,etaInf]
paramsLAB_HB = [nPow,K,tauY0,ts]
paramsLAB_HBMod = [nPow,K,tauY0,ts,eta0]

## Calculations
gammaDot = np.logspace(-2,2,num=100)
t = np.linspace(0,12500,num=100)

# HB

etaHB1 = HB_eta(1e-1,t,*paramsLAB_HB)

tauHB2 = HB_eta(gammaDot,1e2,*paramsLAB_HB)
etaHB2 = HB_eta(gammaDot,1e2,*paramsLAB_HB)

tauHB3 = HB_tau(gammaDot,1e1,*paramsLAB_HB)
etaHB3 = HB_eta(gammaDot,1e1,*paramsLAB_HB)
# HBMod
# tauHBMod = HBMod_tau(gammaDot,1e2,*paramsHBMod)
# etaHBMod = HBMod_eta(gammaDot,1e2,*paramsHBMod)
# SMD

etaSMD1 = SMDMod_eta(1e-1,t,*paramsLAB_SMD)

tauSMD2 = SMDMod_tau(gammaDot,1e2,*paramsLAB_SMD)
etaSMD2 = SMDMod_eta(gammaDot,1e2,*paramsLAB_SMD)

tauSMDLAB = SMDMod_tau(gammaDot,1e2,*paramsLAB_SMD)
tauSMDPWS = SMDMod_tau(gammaDot,1e2,*paramsPWS_SMD)
etaSMDLAB = SMDMod_eta(gammaDot,1e2,*paramsLAB_SMD)
etaSMDPWS = SMDMod_eta(gammaDot,1e2,*paramsPWS_SMD)


# tau_Y = tauY(t,ts,tau0)

## Plotting
# Figures
# fig1,ax1 = plt.subplots()
# fig2,ax2 = plt.subplots()
fig3,ax3 = plt.subplots()

# Plots
# ax1.loglog(t,etaHB1,label='HB')
# # ax1.loglog(t,etaHBMod,label='HBMod')
# ax1.loglog(t,etaSMD1,label='SMDMod')
# ax1.set_xlabel('$t(s)$')
# ax1.set_ylabel('$\eta(Pa.s)$')
# ax1.legend()

# ax2.loglog(gammaDot,etaHB2,label='HB')
# # ax2.loglog(gammaDot,tauHBMod,label='HBMod')
# ax2.loglog(gammaDot,etaSMD2,label='SMDMod')
# ax2.set_xlabel('$\dot{\gamma}(s^{-1})$')
# ax2.set_ylabel(r'$\eta (Pa.s)$')
# ax2.legend()

ax3.loglog(gammaDot,tauSMDLAB,label='LAB Sample Fitted by SMD')
# ax2.loglog(gammaDot,tauHBMod,label='HBMod')
ax3.loglog(gammaDot,tauSMDPWS,label='PWS Sample Fitted by SMD')
ax3.loglog(ExpCurveLAB[:,0],ExpCurveLAB[:,1],'.r', label='Rheometry LAB Sample')
ax3.loglog(ExpCurvePWS[:,0],ExpCurvePWS[:,1],'.k', label='Rheometry PWS Sample')
ax3.set_xlabel('$\dot{\gamma}(s^{-1})$')
ax3.set_ylabel(r'$\tau (Pa)$')
ax3.legend()
ax3.set_xlim([1e-2,1e2])
ax3.set_ylim([1e0,1e2])

# plt.show()
# Saving
# fig1.savefig('./Images/rheoFitting_EtaTime'+detail+figFormat,dpi=400)
# fig2.savefig('./Images/rheoFitting_EtaGammaDot'+detail+figFormat,dpi=400)
fig3.savefig('./Images/rheoFitting_TauGammaDot'+detail+figFormat,dpi=400)