# Load the Pandas libraries with alias 'pd' 
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.font_manager
import numpy as np
from scipy import optimize

# def fit_fun(x, a, b, c):
#     return a * np.exp(-b * x) + c
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
# df = pd.read_csv(".Labeled/20150429.xlsx") 
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


# Read Resampled Experiment simDF
tEnd = 11000
SimShow = False #True # 
rawShow = True # False # 
avgShow = False #True # 
errorShow = False #True # 
every = 1
testDate = '20160114'
Analysis = 'LongCure'
colLength = 4
simFile = testDate+'_pressurePerDepth'
filename = testDate+'_'+Analysis
cmap = cm.get_cmap('jet')
df = pd.read_csv(r'./Resampled/'+filename+'.csv',sep=';')

pressure_Pa=df.iloc[:,[1,12,13,14,15,16]]
pressure_Pa_raw=df.iloc[:,1:12]
fluidLossMassDF=df.iloc[:,17:21]

# Sensor Tags
# fluidLossMass = 'FL_M_2A'
# fluidLossMass_g = pd.concat([df['PASSED_SECONDS'],df[fluidLossMass]],axis=1)
# # Remove decresing or negative Masses
# fluidLossMass_g.loc[fluidLossMass_g[fluidLossMass] < 0,fluidLossMass] = 0
# fluidLossMass_g.loc[fluidLossMass_g[fluidLossMass] < 0,fluidLossMass] = np.max(fluidLossMass_g[fluidLossMass])
# fluidLossStartTime = fluidLossMass_g[fluidLossMass_g[fluidLossMass]==0].PASSED_SECONDS.max()
# columnFillStart = 'P_1B'
# pressureDropStartTime = df[df[columnFillStart]==df[columnFillStart].max()].PASSED_SECONDS.max()

# Passed Seconds Reset
# passed_seconds = df['PASSED_SECONDS']-fluidLossStartTime
# passed_seconds = df['PASSED_SECONDS']-pressureDropStartTime

# psi to Pa Conversion Factor
# psia2Pa = 6894.76
# psig2Pa = psia2Pa-corr
# patmpsi = 14.7
# patmPa = psia2Pa*patmpsi + PZmin
# colLength = 4
# pressureResFile = filename+'_PressurePerDepth';legNCol=5;legFontSize = 12; legOrder = [0,5,1,6,2,7,3,8,4,9]
# pressureResFile = filename+'_PressureWithSim';legNCol=5;legFontSize = 12; legOrder = [0,5,10,1,6,11,2,7,12,3,8,13,4,9,14]
pressureResFile = filename+'_PressurePerExp';legNCol=5;legFontSize = 12; legOrder = [0,1,2,3,4,5,6,7,8,9] #legOrder = [0,1,10,2,3,11,4,5,12,6,7,13,8,9,14]
flowrateResFile = filename+'_Flowrate'


# # Initialize Figures
# fig,ax = plt.subplots(3,3)

# a=1
# # Define Time Series for each pressure sensor\
#     # P_1B;P_1C;FL_M_2A;P_2B;P_2C;P_3B;P_3C;P_4A;P_4B;P_4C;P_5B;P_5C;P_6A;P_6B;P_6C;P_7B;P_7C;P_8A;P_8B;P_8C;P_9B;P_9C
# module_1_sensors = ['P_1B']
# # module_1_depth = pd.Series(colLength-0.2-0*0.4 for _ in range(len(passed_senconds)))
# pressureMod_psi_1 = df[module_1_sensors] + patmpsi
# pressure_Pa_1 = df[module_1_sensors].mean(axis=1)*psig2Pa + patmPa
# # pressure_Pa_1.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_1.plot(fig=fig,ax=ax[0,0]);ax[0,0].legend()

# module_2_sensors = ['P_2C']
# pressureMod_psi_2 = df[module_2_sensors] + patmpsi
# pressure_Pa_2 = df[module_2_sensors].mean(axis=1)*psig2Pa + patmPa
# # pressure_Pa_2.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_2.plot(ax=ax[0,1]);ax[0,1].legend()

# module_3_sensors = ['P_3C']
# pressureMod_psi_3 = df[module_3_sensors] + patmpsi
# pressure_Pa_3 = df[module_3_sensors].mean(axis=1)*psig2Pa + patmPa
# # pressure_Pa_3.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_3.plot(ax=ax[0,2]);ax[0,2].legend()

# # module_4_sensors = ['P_4A','P_4B','P_4C']
# module_4_sensors = ['P_4B','P_4C']
# pressureMod_psi_4 = df[module_4_sensors] + patmpsi
# pressure_Pa_4 = df[module_4_sensors].mean(axis=1)*psig2Pa + patmPa
# # pressure_Pa_4.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_4.plot(ax=ax[1,0]);ax[1,0].legend()

# module_5_sensors = ['P_5B','P_5C']
# pressureMod_psi_5 = df[module_5_sensors] + patmpsi
# pressure_Pa_5 = df[module_5_sensors].mean(axis=1)*psig2Pa + patmPa
# # pressure_Pa_5.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_5.plot(ax=ax[1,1]);ax[1,1].legend()

# module_6_sensors = ['P_6C']
# pressureMod_psi_6 = df[module_6_sensors] + patmpsi
# pressure_Pa_6 = df[module_6_sensors].mean(axis=1)*psig2Pa + patmPa
# # pressure_Pa_6.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_6.plot(ax=ax[1,2]);ax[1,2].legend()

# module_7_sensors = ['P_7B','P_7C']
# pressureMod_psi_7 = df[module_7_sensors] + patmpsi
# pressure_Pa_7 = df[module_7_sensors].mean(axis=1)*psig2Pa + patmPa
# # pressure_Pa_7.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_7.plot(ax=ax[2,0]);ax[2,0].legend()

# module_8_sensors = ['P_8A','P_8B','P_8C']
# pressureMod_psi_8 = df[module_8_sensors] + patmpsi
# pressure_Pa_8 = df[module_8_sensors].mean(axis=1)*psig2Pa + patmPa
# # pressure_Pa_8.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_8.plot(ax=ax[2,1]);ax[2,1].legend()

# module_9_sensors = ['P_9B','P_9C']
# pressureMod_psi_9 = df[module_9_sensors] + patmpsi
# pressure_Pa_9 = df[module_9_sensors].mean(axis=1)*psig2Pa + patmPa
# # pressure_Pa_9.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_9.plot(ax=ax[2,2]);ax[2,2].legend()

# fig2,ax2 = plt.subplots()
# pressure_Pa = pd.concat([passed_seconds,pressure_Pa_5,pressure_Pa_4,pressure_Pa_3,pressure_Pa_2,\
#                         pressure_Pa_1],axis=1)
# # pressure_Pa = pd.concat([passed_seconds,pressure_Pa_1,pressure_Pa_3,\
# #                         pressure_Pa_5,pressure_Pa_7],axis=1)
# columns = ['t(s)']
# for i in range(len(pressure_Pa.columns)-1): #[0,2,4,6]: # 
#     j = len(pressure_Pa.columns) -2 - i
#     # j=i
#     columns.append('$P_e$('+str(np.round(colLength-0.2-j*0.4,1))+'m)')


# UptestDate Columns
# pressure_Pa.columns = columns
interestData = (df['t(s)'] <= tEnd) # & (pressure_Pa['t(s)']>=0)

fig2, ax2 = plt.subplots()
colorsRaw = plt.cm.jet(np.linspace(0,1,5))

## Simulated Results
if SimShow:
    simDF = pd.read_csv('./Simulated/'+simFile+'.csv') #Time(s),Depth(m),Pressure(Pa)
    
    i=0
    maxTime = simDF['Time(s)'].max()
    minTime = 0 #simDF['Time(s)'].min()

    groupedSimDF = simDF.groupby(['Depth(m)'])
    colorsSim = plt.cm.jet(np.linspace(0,1,len(groupedSimDF)))
    for key, grp in groupedSimDF: 
        depthLabel = '$P_s ({:.1f}m)$'.format(key)
        ax = grp.plot(ax=ax2, kind='line', x='Time(s)', y='Pressure(Pa)', \
                        c = colorsSim[i], label=depthLabel)
        i+=1
    plt.legend(loc = 'lower center',bbox_to_anchor=(0.5, 1.01),ncol=6,fontsize = legFontSize)
    
if rawShow: 
    i=0
    k=0
    pressure_Pa_raw = pressure_Pa_raw[interestData]
    
    while i < pressure_Pa_raw.shape[1]-1:
        j = len(pressure_Pa.columns) -2 - k
        i+=1
        ax2.plot(pressure_Pa_raw.iloc[:,0], pressure_Pa_raw.iloc[:,i], marker='^',markeredgecolor= colorsRaw[k],color= 'none',markevery=every,label='$P_{e1}$('+str(np.round(colLength-0.2-j*0.4,1))+'m)'); i+=1 
        ax2.plot(pressure_Pa_raw.iloc[:,0], pressure_Pa_raw.iloc[:,i], marker='o',markeredgecolor= colorsRaw[k],color= 'none',markevery=every,label='$P_{e2}$('+str(np.round(colLength-0.2-j*0.4,1))+'m)')
        # ax2.plot(pressure_Pa_raw.iloc[:,0], pressure_Pa_raw.iloc[:,i], marker='s',markeredgecolor= colorsRaw[k],color= 'none',markevery=every,label='$P_{e3}$('+str(np.round(colLength-0.2-j*0.4,1))+'m)')
        k+=1

    ax2.set_ylabel('$P$(Pa)')
    ax2.set_xlabel('$t$(s)')
    # ax2.set_ylim([120000,230000])
    # pos1 = ax2.get_position() # get the original position 
    # pos2 = [pos1.x0 + 0.03, pos1.y0,  pos1.width, pos1.height] 
    # ax2.set_position(pos2) # set a new position
    ax2.legend(ncol=legNCol,fontsize= legFontSize)
#
i=0
k=0
sigma=0
dataColumns = pressure_Pa_raw.shape[1]-1

while i < dataColumns:
    i+=1
    pressure_Pa_raw['Mean_'+str(k)] = pressure_Pa_raw.iloc[:,i:i+2].mean(axis=1)
    pressure_Pa_raw['Std_'+str(k)] = pressure_Pa_raw.iloc[:,i:i+2].std(axis=1,numeric_only=True);i+=1
    # pressure_Pa_raw['maxStd_'+str(k)] = pressu3re_Pa_raw['Std'].max()
    k+=1
series = k

# plt.rcParams['font.family'] = ['serif']
# plt.rcParams['font.serif'] = ['Times New Roman']
if avgShow:
    # pressure_Pa[interestData].plot(x='t(s)',marker='.',colormap=cmap,linewidth=0,ax=ax2)
    for k in range(series):
        j = len(pressure_Pa.columns) -2 - k
        ax2.plot(pressure_Pa_raw['t(s)'],pressure_Pa_raw['Mean_'+str(k)],marker='.',linewidth=0,
                    label='$P_{avg}$('+str(np.round(colLength-0.2-j*0.4,1))+'m)',color=colorsRaw[k])

if errorShow:
    for k in range(series):
        j = len(pressure_Pa.columns) -2 - k
        ax2.fill_between(pressure_Pa_raw['t(s)'],pressure_Pa_raw['Mean_'+str(k)]+pressure_Pa_raw['Std_'+str(k)],
                                                pressure_Pa_raw['Mean_'+str(k)]-pressure_Pa_raw['Std_'+str(k)],
                                                facecolor=colorsRaw[k], alpha=0.2,label='$P_{Er}$('+str(np.round(colLength-0.2-j*0.4,1))+'m)')
    
ax2.set_ylabel('$P$(Pa)', fontsize=16)
for tick in ax2.xaxis.get_major_ticks():
                tick.label.set_fontsize(14)
for tick in ax2.yaxis.get_major_ticks():
                tick.label.set_fontsize(14) 
ax2.set_xlabel('$t$(s)', fontsize=16)
ax2.set_xlim([0,tEnd])
ax2.set_ylim([90000,175000])
# pos1 = ax2.get_position() # get the original position 
# pos2 = [pos1.x0 + 0.03, pos1.y0,  pos1.width, pos1.height] 
# ax2.set_position(pos2) # set a new position
handles, labels = plt.gca().get_legend_handles_labels()
order = legOrder
ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol=legNCol,fontsize= legFontSize)
# plt.show()
fig2.set_size_inches(10, 7)
fig2.savefig('./Images/'+pressureResFile+'.png',dpi=400)

# # Show Header
# print(presure_pa_1.head(10))

# Pressure(psi)
# presure_Pa = df[sensorTag]*psi2Pa
# presure_Pa.plot(ax = ax[0],label=sensorTag)

# print(df.iloc[0:10])

# Resampled Pressure(g)
# fig3,ax3 = plt.subplots()
# ax4=ax3.twinx()
# HFluidLoss = 0.2                # m
# ROut = 9*0.0254/2               # m
# AOut = 2*np.pi*ROut*HFluidLoss  # m²
# rhoOut = 1000                   # kg/m³
# fluidLossMassDF = pd.concat([passed_seconds,fluidLossMass_g[fluidLossMass]],axis=1)
# fluidLossMassDF.columns=['t(s)','m(g)']
# fluidLossMassDF.drop(fluidLossMassDF[interestData].index, inplace = False)
# fluidLossMassDF['mDot(kg/s)'] = fluidLossMassDF['m(g)'].diff()/fluidLossMassDF['t(s)'].diff()/1000
# fluidLossMassDF.fillna(method = 'bfill',inplace=True)
# fluidLossMassDF['Vr(m/s)']=fluidLossMassDF['mDot(kg/s)']*(1/(rhoOut*AOut)) 
# fluidLossMassPlot = fluidLossMassDF[interestData]
# x_data = fluidLossMassPlot['t(s)']
# y_data = fluidLossMassPlot['mDot(kg/s)']
# # y_data[y_data <= ]
# def fitFunc(t, a, b, c, d, e, ts):
#     ti = 500
#     # return a*t + b
#     # return a*pow(1-b,t)
#     y = (t>=ti)*(a/(pow(3*t,b)) - (1/(c))*np.exp((t-ts)/(d)) + e) + (t<ti)*((4e-5/600)*t)
#     return y
#     # return a/(pow(t,b)) - (1/(c))*np.exp((t-ts)/(d))

# def func(x, a, b, c):

#     return a * np.exp(-b * x) + c

# # Fit 
# a= 5.9 # ^= >
# b=1.62 # < = >
# c=50000 # ^ = ---^
# d=31000
# e=2.85e-5
# ts=1200
# params = [a, b, c, d, e, ts]
# # params, params_covariance = optimize.curve_fit(fitFunc, x_data, y_data, p0=params)
# # params, params_covariance = optimize.curve_fit(func, x_data, y_data, p0=[a,b,c])


# print(params)



# print(fluidLossMassPlot)
# fluidLossMassPlot.plot(kind='scatter',x='t(s)',y='mDot(kg/s)', ax=ax4, color='orange',legend=None)
# fluidLossMassPlot.plot(kind='scatter',x='t(s)',y='m(g)', ax=ax3,  color='blue',legend=None)
# ax4.plot(x_data, fitFunc(x_data, *params), 'k--',\
#          label='fit: a=%5.4g, b=%5.4g, c=%5.4g, d=%5.4g, e=%5.4g, ts=%5.4g' % tuple(params))
# ax3.set_title(testDate)
# ax3.set_ylabel('$m_f$(g)')
# ax4.set_ylabel('$\dot{m}_f$(kg/s)')
# ax4.set_xlim([0,tEnd])
# ax3.set_xlabel('$t$(s)')
# ax3.set_ylim([0,100])
# ax4.set_ylim([0,0.000045])
# # plt.show()
# fig3.savefig('./Images/'+flowrateResFile+'.png',dpi=300)