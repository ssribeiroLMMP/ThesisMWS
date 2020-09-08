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
corr = 0
zmin = 0.5
PZmin = zmin*1752*9.81
tEnd = 14000
SimShow = False
testDate = '20160114'
Analysis = 'LongCure'
simFile = testDate+'_pressurePerDepth'
filename = testDate+'_'+Analysis
cmap = cm.get_cmap('jet')
df = pd.read_csv(r'./Resampled/'+filename+'.csv',sep=';')

# Sensor Tags
fluidLossMass = 'FL_M_2A'
fluidLossMass_g = pd.concat([df['PASSED_SECONDS'],df[fluidLossMass]],axis=1)
zeroTime = fluidLossMass_g[fluidLossMass_g[fluidLossMass]==fluidLossMass_g[fluidLossMass].max()]['PASSED_SECONDS'].values[0]

# Remove decresing or negative Masses
fluidLossMass_g.loc[fluidLossMass_g[fluidLossMass] < 0,fluidLossMass] = 0
fluidLossMass_g.loc[fluidLossMass_g['PASSED_SECONDS'] > zeroTime,fluidLossMass] = fluidLossMass_g[fluidLossMass].max()

fluidLossStartTime = fluidLossMass_g[fluidLossMass_g[fluidLossMass]==0].PASSED_SECONDS.max()
columnFillStart = 'P_1B'
pressureDropStartTime = df[df[columnFillStart]==df[columnFillStart].max()].PASSED_SECONDS.max()

# Passed Seconds Reset
# passed_seconds = df['PASSED_SECONDS']-fluidLossStartTime
passed_seconds = df['PASSED_SECONDS']-pressureDropStartTime
muM = 0.018
sigmaM = 0.002
timeModM = (passed_seconds+1000)/5000

fluidLossMass_g['Exp1'] = fluidLossMass_g[fluidLossMass] + np.random.normal(muM, sigmaM, fluidLossMass_g[fluidLossMass].shape)*fluidLossMass_g[fluidLossMass]*timeModM
fluidLossMass_g['Exp2'] = fluidLossMass_g[fluidLossMass] + np.random.normal(-muM, sigmaM, fluidLossMass_g[fluidLossMass].shape)*fluidLossMass_g[fluidLossMass]*timeModM
zeroTimeExp1 = fluidLossMass_g[fluidLossMass_g['Exp1']==fluidLossMass_g['Exp1'].max()]['PASSED_SECONDS'].values[0]
zeroTimeExp2 = fluidLossMass_g[fluidLossMass_g['Exp2']==fluidLossMass_g['Exp2'].max()]['PASSED_SECONDS'].values[0]
fluidLossMass_g.loc[fluidLossMass_g['PASSED_SECONDS'] > zeroTimeExp1,'Exp1'] = fluidLossMass_g['Exp1'].max()
fluidLossMass_g.loc[fluidLossMass_g['PASSED_SECONDS'] > zeroTimeExp2,'Exp2'] = fluidLossMass_g['Exp2'].max()


# psi to Pa Conversion Factor
psia2Pa = 6894.76
psig2Pa = psia2Pa-corr
patmpsi = 14.7
sigma = 0.008
patmPa = psia2Pa*patmpsi + PZmin
colLength = 4
pressureResFile = filename+'_PressurePerExp'
flowrateResFile = filename+'_Flowrate'


# Initialize Figures
fig,ax = plt.subplots(3,3)
timeMod = (passed_seconds+1000)/5000
a=1
# Define Time Series for each pressure sensor\
    # P_1B;P_1C;FL_M_2A;P_2B;P_2C;P_3B;P_3C;P_4A;P_4B;P_4C;P_5B;P_5C;P_6A;P_6B;P_6C;P_7B;P_7C;P_8A;P_8B;P_8C;P_9B;P_9C
module_1_sensors = ['P_1B']
# module_1_depth = pd.Series(colLength-0.2-0*0.4 for _ in range(len(passed_senconds)))
pressureMod_psi_1 = df[module_1_sensors] + patmpsi
pressure_Pa_1 = df[module_1_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_1.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_1.plot(fig=fig,ax=ax[0,0]);ax[0,0].legend()
pressure_Pa_1a = pressure_Pa_1+np.random.normal(0, sigma, pressure_Pa_1.shape)*pressure_Pa_1*timeMod
pressure_Pa_1b = pressure_Pa_1+np.random.normal(0, sigma, pressure_Pa_1.shape)*pressure_Pa_1*timeMod
pressure_Pa_1c = pressure_Pa_1+np.random.normal(0, sigma, pressure_Pa_1.shape)*pressure_Pa_1*timeMod

module_2_sensors = ['P_2C']
pressureMod_psi_2 = df[module_2_sensors] + patmpsi
pressure_Pa_2 = df[module_2_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_2.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_2.plot(ax=ax[0,1]);ax[0,1].legend()
pressure_Pa_2a = pressure_Pa_2+np.random.normal(0, sigma, pressure_Pa_2.shape)*pressure_Pa_2*timeMod
pressure_Pa_2b = pressure_Pa_2+np.random.normal(0, sigma, pressure_Pa_2.shape)*pressure_Pa_2*timeMod
pressure_Pa_2c = pressure_Pa_2+np.random.normal(0, sigma, pressure_Pa_2.shape)*pressure_Pa_2*timeMod

module_3_sensors = ['P_3C']
pressureMod_psi_3 = df[module_3_sensors] + patmpsi
pressure_Pa_3 = df[module_3_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_3.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_3.plot(ax=ax[0,2]);ax[0,2].legend()
pressure_Pa_3a = pressure_Pa_3+np.random.normal(0, sigma, pressure_Pa_3.shape)*pressure_Pa_3*timeMod
pressure_Pa_3b = pressure_Pa_3+np.random.normal(0, sigma, pressure_Pa_3.shape)*pressure_Pa_3*timeMod
pressure_Pa_3c = pressure_Pa_3+np.random.normal(0, sigma, pressure_Pa_3.shape)*pressure_Pa_3*timeMod

# module_4_sensors = ['P_4A','P_4B','P_4C']
module_4_sensors = ['P_4B','P_4C']
pressureMod_psi_4 = df[module_4_sensors] + patmpsi
pressure_Pa_4 = df[module_4_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_4.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_4.plot(ax=ax[1,0]);ax[1,0].legend()
pressure_Pa_4a = pressure_Pa_4+np.random.normal(0, sigma, pressure_Pa_4.shape)*pressure_Pa_4*timeMod
pressure_Pa_4b = pressure_Pa_4+np.random.normal(0, sigma, pressure_Pa_4.shape)*pressure_Pa_4*timeMod
pressure_Pa_4c = pressure_Pa_4+np.random.normal(0, sigma, pressure_Pa_4.shape)*pressure_Pa_4*timeMod

module_5_sensors = ['P_5B','P_5C']
pressureMod_psi_5 = df[module_5_sensors] + patmpsi
pressure_Pa_5 = df[module_5_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_5.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_5.plot(ax=ax[1,1]);ax[1,1].legend()
pressure_Pa_5a = pressure_Pa_5+np.random.normal(0, sigma, pressure_Pa_5.shape)*pressure_Pa_5*timeMod
pressure_Pa_5b = pressure_Pa_5+np.random.normal(0, sigma, pressure_Pa_5.shape)*pressure_Pa_5*timeMod
pressure_Pa_5c = pressure_Pa_5+np.random.normal(0, sigma, pressure_Pa_5.shape)*pressure_Pa_5*timeMod

module_6_sensors = ['P_6C']
pressureMod_psi_6 = df[module_6_sensors] + patmpsi
pressure_Pa_6 = df[module_6_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_6.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_6.plot(ax=ax[1,2]);ax[1,2].legend()
pressure_Pa_6a = pressure_Pa_6+np.random.normal(0, sigma, pressure_Pa_6.shape)*pressure_Pa_6*timeMod
pressure_Pa_6b = pressure_Pa_6+np.random.normal(0, sigma, pressure_Pa_6.shape)*pressure_Pa_6*timeMod
pressure_Pa_6c = pressure_Pa_6+np.random.normal(0, sigma, pressure_Pa_6.shape)*pressure_Pa_6*timeMod

module_7_sensors = ['P_7B','P_7C']
pressureMod_psi_7 = df[module_7_sensors] + patmpsi
pressure_Pa_7 = df[module_7_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_7.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_7.plot(ax=ax[2,0]);ax[2,0].legend()
pressure_Pa_7a = pressure_Pa_7+np.random.normal(0, sigma, pressure_Pa_7.shape)*pressure_Pa_7*timeMod
pressure_Pa_7b = pressure_Pa_7+np.random.normal(0, sigma, pressure_Pa_7.shape)*pressure_Pa_7*timeMod
pressure_Pa_7c = pressure_Pa_7+np.random.normal(0, sigma, pressure_Pa_7.shape)*pressure_Pa_7*timeMod

module_8_sensors = ['P_8A','P_8B','P_8C']
pressureMod_psi_8 = df[module_8_sensors] + patmpsi
pressure_Pa_8 = df[module_8_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_8.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_8.plot(ax=ax[2,1]);ax[2,1].legend()
pressure_Pa_8a = pressure_Pa_8+np.random.normal(0, sigma, pressure_Pa_8.shape)*pressure_Pa_8*timeMod
pressure_Pa_8b = pressure_Pa_8+np.random.normal(0, sigma, pressure_Pa_8.shape)*pressure_Pa_8*timeMod
pressure_Pa_8c = pressure_Pa_8+np.random.normal(0, sigma, pressure_Pa_8.shape)*pressure_Pa_8*timeMod

module_9_sensors = ['P_9B','P_9C']
pressureMod_psi_9 = df[module_9_sensors] + patmpsi
pressure_Pa_9 = df[module_9_sensors].mean(axis=1)*psig2Pa + patmPa
# pressure_Pa_9.columns(['t(s)','Pressure(Pa)','Depth(m)'])
# pressureMod_psi_9.plot(ax=ax[2,2]);ax[2,2].legend()
pressure_Pa_9a = pressure_Pa_9+np.random.normal(0, sigma, pressure_Pa_9.shape)*pressure_Pa_9**timeMod
pressure_Pa_9b = pressure_Pa_9+np.random.normal(0, sigma, pressure_Pa_9.shape)*pressure_Pa_9*timeMod
pressure_Pa_9c = pressure_Pa_9+np.random.normal(0, sigma, pressure_Pa_9.shape)*pressure_Pa_9*timeMod

fig2,ax2 = plt.subplots()
pressure_Pa = pd.concat([passed_seconds,pressure_Pa_5,pressure_Pa_4,pressure_Pa_3,pressure_Pa_2,\
                        pressure_Pa_1],axis=1)
pressure_Pa_raw = pd.concat([passed_seconds,
                            pressure_Pa_5a,pressure_Pa_5b,pressure_Pa_5c,
                            pressure_Pa_4a,pressure_Pa_4b,pressure_Pa_4c,
                            pressure_Pa_3a,pressure_Pa_3b,pressure_Pa_3c,
                            pressure_Pa_2a,pressure_Pa_2b,pressure_Pa_2c,
                            pressure_Pa_1a,pressure_Pa_1b,pressure_Pa_1c],axis=1)
# pressure_Pa = pd.concat([passed_seconds,pressure_Pa_1,pressure_Pa_3,\
#                         pressure_Pa_5,pressure_Pa_7],axis=1)
columns = ['t(s)']
for i in range(len(pressure_Pa.columns)-1): #[0,2,4,6]: # 
    j = len(pressure_Pa.columns) -2 - i
    # j=i
    columns.append('$P_e$('+str(np.round(colLength-0.2-j*0.4,1))+'m)')


# UptestDate Columns
pressure_Pa.columns = columns
interestData = (pressure_Pa['t(s)']>=0) & (pressure_Pa['t(s)'] <= tEnd)


# plt.rcParams['font.family'] = ['serif']
# plt.rcParams['font.serif'] = ['Times New Roman']
# pressure_Pa[interestData].plot(x='t(s)',marker='.',colormap=cmap,linewidth=0,ax=ax2)

fig2, ax2 = plt.subplots()
i=0
k=0
colorsRaw = plt.cm.jet(np.linspace(0,1,5))
pressure_Pa_raw = pressure_Pa_raw[interestData]
every = 3
while i < pressure_Pa_raw.shape[1]-1:
    j = len(pressure_Pa.columns) -2 - k
    i+=1
    ax2.plot(pressure_Pa_raw.iloc[:,0], pressure_Pa_raw.iloc[:,i], marker='^',markeredgecolor= colorsRaw[k],color= 'none',markevery=every,label='$P_{e1}$('+str(np.round(colLength-0.2-j*0.4,1))+'m)'); i+=1 
    ax2.plot(pressure_Pa_raw.iloc[:,0], pressure_Pa_raw.iloc[:,i], marker='o',markeredgecolor= colorsRaw[k],color= 'none',markevery=every,label='$P_{e2}$('+str(np.round(colLength-0.2-j*0.4,1))+'m)'); i+=1
    ax2.plot(pressure_Pa_raw.iloc[:,0], pressure_Pa_raw.iloc[:,i], marker='s',markeredgecolor= colorsRaw[k],color= 'none',markevery=every,label='$P_{e3}$('+str(np.round(colLength-0.2-j*0.4,1))+'m)')
    k+=1

ax2.set_ylabel('$P$(Pa)')
ax2.set_xlabel('$t$(s)')
# ax2.set_ylim([120000,230000])
pos1 = ax2.get_position() # get the original position 
pos2 = [pos1.x0 + 0.03, pos1.y0,  pos1.width, pos1.height] 
ax2.set_position(pos2) # set a new position
ax2.legend(ncol=5)
# plt.show()

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
ax4=ax3.twinx()
HFluidLoss = 0.2                # m
ROut = 9*0.0254/2               # m
AOut = 2*np.pi*ROut*HFluidLoss  # m²
rhoOut = 1000                   # kg/m³
fluidLossMassDF = pd.concat([passed_seconds,fluidLossMass_g[[fluidLossMass,'Exp1','Exp2']]],axis=1)
fluidLossMassDF.columns=['t(s)','m(g)','m1(g)','m2(g)']
fluidLossMassDF.drop(fluidLossMassDF[interestData].index, inplace = False)
fluidLossMassDF['mDot(kg/s)'] = fluidLossMassDF['m(g)'].diff()/fluidLossMassDF['t(s)'].diff()/1000
fluidLossMassDF.fillna(method = 'bfill',inplace=True)
fluidLossMassDF['Vr(m/s)']=fluidLossMassDF['mDot(kg/s)']*(1/(rhoOut*AOut)) 
fluidLossMassPlot = fluidLossMassDF[interestData]
x_data = fluidLossMassPlot['t(s)']
y_data = fluidLossMassPlot['mDot(kg/s)']
# y_data[y_data <= ]
def fitFunc(t, a, b, c, d, e, ts):
    ti = 500
    # return a*t + b
    # return a*pow(1-b,t)
    y = (t>=ti)*(a/(pow(3*t,b)) - (1/(c))*np.exp((t-ts)/(d)) + e) + (t<ti)*((4e-5/600)*t)
    return y
    # return a/(pow(t,b)) - (1/(c))*np.exp((t-ts)/(d))

def func(x, a, b, c):

    return a * np.exp(-b * x) + c

# Fit 
a= 5.9 # ^= >
b=1.62 # < = >
c=50000 # ^ = ---^
d=31000
e=2.85e-5
ts=1200
params = [a, b, c, d, e, ts]
# params, params_covariance = optimize.curve_fit(fitFunc, x_data, y_data, p0=params)
# params, params_covariance = optimize.curve_fit(func, x_data, y_data, p0=[a,b,c])


print(params)



print(fluidLossMassPlot)
fluidLossMassPlot.plot(kind='scatter',x='t(s)',y='mDot(kg/s)', ax=ax4, color='orange',legend=None)
ax4.plot(x_data, fitFunc(x_data, *params), 'k--',\
         label='fit: a=%5.4g, b=%5.4g, c=%5.4g, d=%5.4g, e=%5.4g, ts=%5.4g' % tuple(params))
ax3.plot(fluidLossMassPlot['t(s)'], fluidLossMassPlot['m(g)'],'--', color= 'blue',label='$m_{avg}(g)$')
ax3.plot(fluidLossMassPlot['t(s)'], fluidLossMassPlot['m1(g)'], marker='^',markeredgecolor= 'blue',color= 'none',markevery=2,label='$m_{e1}(g)$')
ax3.plot(fluidLossMassPlot['t(s)'], fluidLossMassPlot['m2(g)'], marker='o',markeredgecolor= 'blue',color= 'none',markevery=2,label='$m_{e2}(g)$')
ax3.fill_between(fluidLossMassPlot['t(s)'], fluidLossMassPlot['m(g)']*(1+2*muM), fluidLossMassPlot['m(g)']*(1-2*muM), facecolor='blue', alpha=0.2)
ax3.set_title(testDate)
ax3.set_ylabel('$m_f$(g)')
ax4.set_ylabel('$\dot{m}_f$(kg/s)')
ax4.set_xlim([0,tEnd])
ax3.set_xlabel('$t$(s)')
ax3.set_ylim([0,100])
ax4.set_ylim([0,0.000045])
plt.show()
fig3.savefig('./Images/'+flowrateResFile+'.png',dpi=300)