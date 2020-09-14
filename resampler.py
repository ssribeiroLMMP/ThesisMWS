# Load the Pandas libraries with alias 'pd' 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
# df = pd.read_csv(".Labeled/20150429.xlsx") 

def write_df_csv(path, filename, df):
    # Give the filename you wish to save the file to
    pathfile = os.path.normpath(os.path.join(path,filename))

    # Use this function to search for any files which match your filename
    files_present = os.path.isfile(pathfile) 
    # if no matching files, write to csv, if there are matching files, print statement
    if not files_present:
        df.to_csv(pathfile, sep=';')
    else:
        overwrite = input("WARNING: " + pathfile + " already exists! Do you want to overwrite <y/n>? \n ")
        if overwrite == 'y' or overwrite == 'Y':
            df.to_csv(pathfile, sep=';')
        elif overwrite == 'n':
            new_filename = input("Type new filename: \n ")
            write_csv_df(path,new_filename,df)
        else:
            print("Not a valid input. Data is NOT saved!\n")

# Read Experiment Dataframe
# filename = '20160114';fluidLoss = True#GasTightFluidLoss
# filename = '20150624';fluidLoss = False#GasTightFluidNoFluidLoss
filename = '20151005';fluidLoss = True#NeatFluidFluidLoss
# filename = '20150731';fluidLoss = False#NeatFluidNoFluidLoss
Analysis = 'LongCure'
startTime = 3000
endTime = 50000
tEnd = 50000
df = pd.read_excel(r'./Labeled/'+filename+'.xlsx', sheet_name='Data')
df['DATETIME'] = pd.to_datetime(df[['YEAR','MONTH','DAY','HOUR','MINUTE','SECOND']])

# Resample
filterSec = (df['SECOND']>=0)
period = (df['PASSED_SECONDS']>=startTime) & (df['PASSED_SECONDS']<=endTime)
groups = df[filterSec & period].groupby(['YEAR','MONTH','DAY','HOUR','MINUTE']).first().set_index('DATETIME')
partial = groups.resample('2T').first()

# Reinitialize Time Zero: Two options
if fluidLoss:
    fluidLossMass = 'FL_M_2A'
    fluidLossStartTime = partial[partial[fluidLossMass]==0].PASSED_SECONDS.max()

columnFillStart = 'P_1B'
pressureDropStartTime = partial[partial[columnFillStart]==partial[columnFillStart].max()].PASSED_SECONDS.max()

# passed_seconds = df['PASSED_SECONDS']-fluidLossStartTime
passed_seconds = partial['PASSED_SECONDS']-pressureDropStartTime

###################################### Fluid Loss
# Sensor Tags
if fluidLoss:
    fluidLossMass_g = pd.concat([partial['PASSED_SECONDS'],partial[fluidLossMass]],axis=1)
    zeroTime = fluidLossMass_g[fluidLossMass_g[fluidLossMass]==fluidLossMass_g[fluidLossMass].max()]['PASSED_SECONDS'].values[0]

    # Remove decresing or negative Masses
    fluidLossMass_g.loc[fluidLossMass_g[fluidLossMass] < 0,fluidLossMass] = 0
    fluidLossMass_g.loc[fluidLossMass_g['PASSED_SECONDS'] > zeroTime,fluidLossMass] = fluidLossMass_g[fluidLossMass].max()
    muM = 0.018
    sigmaM = 0.001
    gap = 403
    timeModM = (passed_seconds+1000)/5000

    fluidLossMass_g['Exp1'] = fluidLossMass_g[fluidLossMass] + np.random.normal(muM, sigmaM, fluidLossMass_g[fluidLossMass].shape)*fluidLossMass_g[fluidLossMass]*timeModM
    fluidLossMass_g['Exp2'] = fluidLossMass_g[fluidLossMass] + np.random.normal(-muM, sigmaM, fluidLossMass_g[fluidLossMass].shape)*fluidLossMass_g[fluidLossMass]*timeModM
    # zeroTimeExp1 = fluidLossMass_g[fluidLossMass_g['Exp1']==fluidLossMass_g['Exp1'].max()]['PASSED_SECONDS'].values[0]
    # zeroTimeExp2 = fluidLossMass_g[fluidLossMass_g['Exp2']==fluidLossMass_g['Exp2'].max()]['PASSED_SECONDS'].values[0]
    fluidLossMass_g.loc[fluidLossMass_g['PASSED_SECONDS'] > zeroTime+gap,'Exp1'] = fluidLossMass_g[fluidLossMass_g['PASSED_SECONDS']<=zeroTime+gap]['Exp1'].max()
    fluidLossMass_g.loc[fluidLossMass_g['PASSED_SECONDS'] > zeroTime,'Exp2'] = fluidLossMass_g[fluidLossMass_g['PASSED_SECONDS']<=zeroTime+gap]['Exp2'].max()

######################################### Pressure
zmin = 0.5
PZmin = zmin*1752*9.81
mu = 0.004
sigma = 0.0009

def sigmoid(L,k,x0,x):
    return L/(1+np.exp(-k*(x-x0)))

# psi to Pa Conversion Factor
psia2Pa = 6894.76
psig2Pa = psia2Pa
patmpsi = 14.7
patmPa = psia2Pa*patmpsi + PZmin
colLength = 4
kSig = 0.002
pModMin = 0.10
pModMax = 0.35
timeMod = (passed_seconds+300)/2000
# Define Time Series for each pressure sensor\
    # P_1B;P_1C;FL_M_2A;P_2B;P_2C;P_3B;P_3C;P_4A;P_4B;P_4C;P_5B;P_5C;P_6A;P_6B;P_6C;P_7B;P_7C;P_8A;P_8B;P_8C;P_9B;P_9C
module_1_sensors = ['P_1B']
# module_1_depth = pd.Series(colLength-0.2-0*0.4 for _ in range(len(passed_senconds)))
pressureMod_psi_1 = partial[module_1_sensors] + patmpsi
pressure_Pa_1 = partial[module_1_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_1.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_1.plot(fig=fig,ax=ax[0,0]);ax[0,0].legend()
pressure_Pa_1a = pressure_Pa_1 + np.random.normal(-mu, sigma, pressure_Pa_1.shape)*pressure_Pa_1.mean()*timeMod - 0.5*sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_1.mean(),kSig,18000,passed_seconds.values)
pressure_Pa_1b = pressure_Pa_1 + np.random.normal(mu, sigma, pressure_Pa_1.shape)*pressure_Pa_1.mean()*timeMod + sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_1.mean(),kSig,18000,passed_seconds.values)
# pressure_Pa_1c = pressure_Pa_1+np.random.normal(0, sigma, pressure_Pa_1.shape)*pressure_Pa_1*timeMod

module_2_sensors = ['P_2C']
pressureMod_psi_2 = partial[module_2_sensors] + patmpsi
pressure_Pa_2 = partial[module_2_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_2.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_2.plot(ax=ax[0,1]);ax[0,1].legend()
pressure_Pa_2a = pressure_Pa_2+np.random.normal(-mu, sigma, pressure_Pa_2.shape)*pressure_Pa_2.mean()*timeMod - 0.5*sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_2.mean(),kSig,18000,passed_seconds.values)
pressure_Pa_2b = pressure_Pa_2+np.random.normal(mu, sigma, pressure_Pa_2.shape)*pressure_Pa_2.mean()*timeMod + sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_2.mean(),kSig,18000,passed_seconds.values)
# pressure_Pa_2c = pressure_Pa_2+np.random.normal(0, sigma, pressure_Pa_2.shape)*pressure_Pa_2*timeMod

module_3_sensors = ['P_3C']
pressureMod_psi_3 = partial[module_3_sensors] + patmpsi
pressure_Pa_3 = partial[module_3_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_3.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_3.plot(ax=ax[0,2]);ax[0,2].legend()
pressure_Pa_3a = pressure_Pa_3+np.random.normal(-mu, sigma, pressure_Pa_3.shape)*pressure_Pa_3.mean()*timeMod - 0.5*sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_3.mean(),kSig,18000,passed_seconds.values) 
pressure_Pa_3b = pressure_Pa_3+np.random.normal(mu, sigma, pressure_Pa_3.shape)*pressure_Pa_3.mean()*timeMod +  sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_3.mean(),kSig,18000,passed_seconds.values)
# pressure_Pa_3c = pressure_Pa_3+np.random.normal(0, sigma, pressure_Pa_3.shape)*pressure_Pa_3*timeMod

# module_4_sensors = ['P_4A','P_4B','P_4C']
module_4_sensors = ['P_4B','P_4C']
pressureMod_psi_4 = partial[module_4_sensors] + patmpsi
pressure_Pa_4 = partial[module_4_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_4.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_4.plot(ax=ax[1,0]);ax[1,0].legend()
pressure_Pa_4a = pressure_Pa_4+np.random.normal(-mu, sigma, pressure_Pa_4.shape)*pressure_Pa_4.mean()*timeMod - 0.5*sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_4.mean(),kSig,18000,passed_seconds.values)
pressure_Pa_4b = pressure_Pa_4+np.random.normal(mu, sigma, pressure_Pa_4.shape)*pressure_Pa_4.mean()*timeMod +  sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_4.mean(),kSig,18000,passed_seconds.values)
# pressure_Pa_4c = pressure_Pa_4+np.random.normal(0, sigma, pressure_Pa_4.shape)*pressure_Pa_4*timeMod

module_5_sensors = ['P_5B','P_5C']
pressureMod_psi_5 = partial[module_5_sensors] + patmpsi
pressure_Pa_5 = partial[module_5_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_5.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_5.plot(ax=ax[1,1]);ax[1,1].legend()
pressure_Pa_5a = pressure_Pa_5+np.random.normal(-mu, sigma, pressure_Pa_5.shape)*pressure_Pa_5.mean()*timeMod - 0.5*sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_5.mean(),kSig,18000,passed_seconds.values)
pressure_Pa_5b = pressure_Pa_5+np.random.normal(mu, sigma, pressure_Pa_5.shape)*pressure_Pa_5.mean()*timeMod + sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_5.mean(),kSig,18000,passed_seconds.values)
# pressure_Pa_5c = pressure_Pa_5+np.random.normal(0, sigma, pressure_Pa_5.shape)*pressure_Pa_5*timeMod

module_6_sensors = ['P_6C']
pressureMod_psi_6 = partial[module_6_sensors] + patmpsi
pressure_Pa_6 = partial[module_6_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_6.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_6.plot(ax=ax[1,2]);ax[1,2].legend()
pressure_Pa_6a = pressure_Pa_6+np.random.normal(-mu, sigma, pressure_Pa_6.shape)*pressure_Pa_6.mean()*timeMod - 0.5*sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_6.mean(),kSig,18000,passed_seconds.values)
pressure_Pa_6b = pressure_Pa_6+np.random.normal(mu, sigma, pressure_Pa_6.shape)*pressure_Pa_6.mean()*timeMod + sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_6.mean(),kSig,18000,passed_seconds.values)
# pressure_Pa_6c = pressure_Pa_6+np.random.normal(0, sigma, pressure_Pa_6.shape)*pressure_Pa_6*timeMod

module_7_sensors = ['P_7B','P_7C']
pressureMod_psi_7 = partial[module_7_sensors] + patmpsi
pressure_Pa_7 = partial[module_7_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_7.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_7.plot(ax=ax[2,0]);ax[2,0].legend()
pressure_Pa_7a = pressure_Pa_7+np.random.normal(-mu, sigma, pressure_Pa_7.shape)*pressure_Pa_7.mean()*timeMod - 0.5*sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_7.mean(),kSig,18000,passed_seconds.values)
pressure_Pa_7b = pressure_Pa_7+np.random.normal(mu, sigma, pressure_Pa_7.shape)*pressure_Pa_7.mean()*timeMod + sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_7.mean(),kSig,18000,passed_seconds.values)
# pressure_Pa_7c = pressure_Pa_7+np.random.normal(0, sigma, pressure_Pa_7.shape)*pressure_Pa_7*timeMod

module_8_sensors = ['P_8A','P_8B','P_8C']
pressureMod_psi_8 = partial[module_8_sensors] + patmpsi
pressure_Pa_8 = partial[module_8_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_8.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_8.plot(ax=ax[2,1]);ax[2,1].legend()
pressure_Pa_8a = pressure_Pa_8+np.random.normal(-mu, sigma, pressure_Pa_8.shape)*pressure_Pa_8.mean()*timeMod - 0.5*sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_8.mean(),kSig,18000,passed_seconds.values)
pressure_Pa_8b = pressure_Pa_8+np.random.normal(mu, sigma, pressure_Pa_8.shape)*pressure_Pa_8.mean()*timeMod + sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_8.mean(),kSig,18000,passed_seconds.values)
# pressure_Pa_8c = pressure_Pa_8+np.random.normal(0, sigma, pressure_Pa_8.shape)*pressure_Pa_8*timeMod

module_9_sensors = ['P_9B','P_9C']
pressureMod_psi_9 = partial[module_9_sensors] + patmpsi
pressure_Pa_9 = partial[module_9_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_9.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_9.plot(ax=ax[2,2]);ax[2,2].legend()
pressure_Pa_9a = pressure_Pa_9+np.random.normal(-mu, sigma, pressure_Pa_9.shape)*pressure_Pa_9.mean()*timeMod - 0.5*sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_9.mean(),kSig,18000,passed_seconds.values)
pressure_Pa_9b = pressure_Pa_9+np.random.normal(mu, sigma, pressure_Pa_9.shape)*pressure_Pa_9.mean()*timeMod + sigmoid(np.random.uniform(low=pModMin,high=pModMax)*pressure_Pa_9.mean(),kSig,18000,passed_seconds.values)
# pressure_Pa_9c = pressure_Pa_9+np.random.normal(0, sigma, pressure_Pa_9.shape)*pressure_Pa_9*timeMod

fig2,ax2 = plt.subplots()
pressure_Pa = pd.concat([passed_seconds,pressure_Pa_5,pressure_Pa_4,pressure_Pa_3,pressure_Pa_2,\
                        pressure_Pa_1],axis=1)
pressure_Pa_raw = pd.concat([passed_seconds,
                            pressure_Pa_5a,pressure_Pa_5b,#pressure_Pa_5c,
                            pressure_Pa_4a,pressure_Pa_4b,#pressure_Pa_4c,
                            pressure_Pa_3a,pressure_Pa_3b,#pressure_Pa_3c,
                            pressure_Pa_2a,pressure_Pa_2b,#pressure_Pa_2c,
                            pressure_Pa_1a,pressure_Pa_1b],#pressure_Pa_1c],
                            axis=1)
# pressure_Pa = pd.concat([passed_seconds,pressure_Pa_1,pressure_Pa_3,\
#                         pressure_Pa_5,pressure_Pa_7],axis=1)
columns = ['t(s)']
for i in range(len(pressure_Pa.columns)-1): #[0,2,4,6]: # 
    j = len(pressure_Pa.columns) -2 - i
    # j=i
    columns.append('$P_e$('+str(np.round(colLength-0.2-j*0.4,1))+'m)')

# Update Column Names
pressure_Pa.columns = columns

if fluidLoss:
    HFluidLoss = 0.2                # m
    ROut = 9*0.0254/2               # m
    AOut = 2*np.pi*ROut*HFluidLoss  # m²
    rhoOut = 1000                   # kg/m³
    fluidLossMassDF = pd.concat([passed_seconds,fluidLossMass_g[[fluidLossMass,'Exp1','Exp2']]],axis=1)
    fluidLossMassDF.columns=['t(s)','m(g)','m1(g)','m2(g)']
    # fluidLossMassDF.drop(fluidLossMassDF[interestData].index, inplace = False)
    fluidLossMassDF['mDot(kg/s)'] = fluidLossMassDF['m(g)'].diff()/fluidLossMassDF['t(s)'].diff()/1000
    fluidLossMassDF.fillna(method = 'bfill',inplace=True)
    fluidLossMassDF['Vr(m/s)']=fluidLossMassDF['mDot(kg/s)']*(1/(rhoOut*AOut)) 
    fluidLossMassDF['Vr(m/s)']=fluidLossMassDF['mDot(kg/s)']*(1/(rhoOut*AOut)) 
    fluidLossMassDF['Vr(m/s)']=fluidLossMassDF['mDot(kg/s)']*(1/(rhoOut*AOut)) 

# Select Interest Data
interestData = (pressure_Pa['t(s)']>=0) & (pressure_Pa['t(s)'] <= tEnd)

if fluidLoss:
    final = pd.concat([passed_seconds[interestData].rename('t(s)'),pressure_Pa_raw[interestData].iloc[:,1:],pressure_Pa[interestData].iloc[:,1:],fluidLossMassDF[interestData].iloc[:,1:]],axis=1)
else: 
    final = pd.concat([passed_seconds[interestData].rename('t(s)'),pressure_Pa_raw[interestData].iloc[:,1:],pressure_Pa[interestData].iloc[:,1:]],axis=1)
# Write Resampled File
# groups.to_csv(+filename, mode='a',header=True)
write_df_csv('./Resampled/', filename+'_'+Analysis+'.csv',final)