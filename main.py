# Load the Pandas libraries with alias 'pd' 
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
# from scipy.optimize import curve_fit

# def fit_fun(x, a, b, c):
#     return a * np.exp(-b * x) + c
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
# df = pd.read_csv(".Labeled/20150429.xlsx") 

# Read Resampled Experiment Dataframe
corr=0
zmin = 4
PZmin = zmin*1800*9.81
filename = '20151005_LongCure'
cmap = cm.get_cmap('jet')
df = pd.read_csv(r'./Resampled/'+filename+'.csv',sep=';')

# Sensor Tags
fluidLossMass = 'FL_M_2A'
fluidLossMass_g = pd.concat([df['PASSED_SECONDS'],df[fluidLossMass]],axis=1)
fluidLossStartTime = fluidLossMass_g[fluidLossMass_g[fluidLossMass]==0].PASSED_SECONDS.max()

# Passed Seconds Reset
passed_seconds = df['PASSED_SECONDS']-fluidLossStartTime


# psi to Pa Conversion Factor
psia2Pa = 6894.76
psig2Pa = psia2Pa-corr
patmpsi = 14.7
patmPa = psia2Pa*patmpsi + PZmin
colLength = 4
pressureResFile = filename+'_PressurePerDepth'
flowrateResFile = filename+'_Flowrate'


# Initialize Figures
fig,ax = plt.subplots(3,3)

a=1
# Define Time Series for each pressure sensor\
    # P_1B;P_1C;FL_M_2A;P_2B;P_2C;P_3B;P_3C;P_4A;P_4B;P_4C;P_5B;P_5C;P_6A;P_6B;P_6C;P_7B;P_7C;P_8A;P_8B;P_8C;P_9B;P_9C
module_1_sensors = ['P_1B','P_1C']
# module_1_depth = pd.Series(colLength-0.2-0*0.4 for _ in range(len(passed_senconds)))
pressureMod_psi_1 = df[module_1_sensors] + patmpsi
pressure_Pa_1 = df[module_1_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_1.columns(['t(s)','Pressure(Pa)','Depth(m)'])
pressureMod_psi_1.plot(fig=fig,ax=ax[0,0]);ax[0,0].legend()

module_2_sensors = ['P_2C']
pressureMod_psi_2 = df[module_2_sensors] + patmpsi
pressure_Pa_2 = df[module_2_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_2.columns(['t(s)','Pressure(Pa)','Depth(m)'])
pressureMod_psi_2.plot(ax=ax[0,1]);ax[0,1].legend()

module_3_sensors = ['P_3C']
pressureMod_psi_3 = df[module_3_sensors] + patmpsi
pressure_Pa_3 = df[module_3_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_3.columns(['t(s)','Pressure(Pa)','Depth(m)'])
pressureMod_psi_3.plot(ax=ax[0,2]);ax[0,2].legend()

# module_4_sensors = ['P_4A','P_4B','P_4C']
module_4_sensors = ['P_4B','P_4C']
pressureMod_psi_4 = df[module_4_sensors] + patmpsi
pressure_Pa_4 = df[module_4_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_4.columns(['t(s)','Pressure(Pa)','Depth(m)'])
pressureMod_psi_4.plot(ax=ax[1,0]);ax[1,0].legend()

module_5_sensors = ['P_5B','P_5C']
pressureMod_psi_5 = df[module_5_sensors] + patmpsi
pressure_Pa_5 = df[module_5_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_5.columns(['t(s)','Pressure(Pa)','Depth(m)'])
pressureMod_psi_5.plot(ax=ax[1,1]);ax[1,1].legend()

module_6_sensors = ['P_6C']
pressureMod_psi_6 = df[module_6_sensors] + patmpsi
pressure_Pa_6 = df[module_6_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_6.columns(['t(s)','Pressure(Pa)','Depth(m)'])
pressureMod_psi_6.plot(ax=ax[1,2]);ax[1,2].legend()

module_7_sensors = ['P_7B','P_7C']
pressureMod_psi_7 = df[module_7_sensors] + patmpsi
pressure_Pa_7 = df[module_7_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_7.columns(['t(s)','Pressure(Pa)','Depth(m)'])
pressureMod_psi_7.plot(ax=ax[2,0]);ax[2,0].legend()

module_8_sensors = ['P_8A','P_8B','P_8C']
pressureMod_psi_8 = df[module_8_sensors] + patmpsi
pressure_Pa_8 = df[module_8_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_8.columns(['t(s)','Pressure(Pa)','Depth(m)'])
pressureMod_psi_8.plot(ax=ax[2,1]);ax[2,1].legend()

module_9_sensors = ['P_9B','P_9C']
pressureMod_psi_9 = df[module_9_sensors] + patmpsi
pressure_Pa_9 = df[module_9_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_9.columns(['t(s)','Pressure(Pa)','Depth(m)'])
pressureMod_psi_9.plot(ax=ax[2,2]);ax[2,2].legend()

fig2,ax2 = plt.subplots()
# pressure_Pa = pd.concat([passed_seconds,pressure_Pa_1,pressure_Pa_2,pressure_Pa_3,pressure_Pa_4,\
#                         pressure_Pa_5,pressure_Pa_6,pressure_Pa_7,pressure_Pa_8,pressure_Pa_9],axis=1)
pressure_Pa = pd.concat([passed_seconds,pressure_Pa_3,\
                        pressure_Pa_5,pressure_Pa_7,pressure_Pa_9],axis=1)
columns = ['t(s)']
for i in [2,4,6,8]:
    columns.append(str(np.round(colLength-0.2-i*0.4+zmin,1))+'m')


# Update Columns
pressure_Pa.columns = columns
interestData = (pressure_Pa['t(s)']>=0) & (pressure_Pa['t(s)'] <= 14000)


pressure_Pa[interestData].plot(x='t(s)',colormap=cmap,ax=ax2)
ax2.set_ylabel('Pressure(Pa)')
pos1 = ax2.get_position() # get the original position 
pos2 = [pos1.x0 + 0.03, pos1.y0,  pos1.width, pos1.height] 
ax2.set_position(pos2) # set a new position
# plt.show()
fig2.savefig('./Images/'+pressureResFile+'.png',dpi=300)

# # Show Header
# print(presure_pa_1.head(10))

# Pressure(psi)
# presure_Pa = df[sensorTag]*psi2Pa
# presure_Pa.plot(ax = ax[0],label=sensorTag)

# print(df.iloc[0:10])

# Resampled Pressure(g)
fig3,ax3 = plt.subplots()
fluidLossMassDF = pd.concat([passed_seconds,fluidLossMass_g[fluidLossMass]],axis=1)
fluidLossMassDF.columns=['t(s)','m(g)']
fluidLossMassDF[interestData].plot(kind='scatter',x='t(s)',y='m(g)', ax=ax3, label=fluidLossMass,color='blue')
# fig3.show()
fig3.savefig('./Images/'+flowrateResFile+'.png',dpi=300)
