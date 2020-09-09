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
fluidLossMassDF=pd.concat([df.iloc[:,1],df.iloc[:,17:21]],axis=1)

## Pressure File Name
# pressureResFile = filename+'_PressurePerDepth';legNCol=5;legFontSize = 12; legOrder = [0,5,1,6,2,7,3,8,4,9]
# pressureResFile = filename+'_PressureWithSim';legNCol=5;legFontSize = 12; legOrder = [0,5,10,1,6,11,2,7,12,3,8,13,4,9,14]
pressureResFile = filename+'_PressurePerExp';legNCol=5;legFontSize = 12; legOrder = [0,1,2,3,4,5,6,7,8,9] #legOrder = [0,1,10,2,3,11,4,5,12,6,7,13,8,9,14]
flowrateResFile = filename+'_ExpFluidLossMass'
flowrateResFile2 = filename+'_FluiLossFlowrateFit'

## FLowrate Fale Name
# # Initialize Figures
# fig,ax = plt.subplots(3,3)

# Interest Data
interestData = (df['t(s)'] <= tEnd) # & (pressure_Pa['t(s)']>=0)

fig2, ax2 = plt.subplots()
colorsRaw = plt.cm.jet(np.linspace(0,1,5))

## Simulated Pressure Results
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
    
# Raw Experimental Results
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

# Calculate Experimental Mean and Standard Deviation
while i < dataColumns:
    i+=1
    pressure_Pa_raw['Mean_'+str(k)] = pressure_Pa_raw.iloc[:,i:i+2].mean(axis=1)
    pressure_Pa_raw['Std_'+str(k)] = pressure_Pa_raw.iloc[:,i:i+2].std(axis=1,numeric_only=True);i+=1
    # pressure_Pa_raw['maxStd_'+str(k)] = pressu3re_Pa_raw['Std'].max()
    k+=1
series = k

# Plot Experimental Mean values
if avgShow:
    # pressure_Pa[interestData].plot(x='t(s)',marker='.',colormap=cmap,linewidth=0,ax=ax2)
    for k in range(series):
        j = len(pressure_Pa.columns) -2 - k
        ax2.plot(pressure_Pa_raw['t(s)'],pressure_Pa_raw['Mean_'+str(k)],marker='.',linewidth=0,
                    label='$P_{avg}$('+str(np.round(colLength-0.2-j*0.4,1))+'m)',color=colorsRaw[k])

# Plot shaded Experimental Std
if errorShow:
    for k in range(series):
        j = len(pressure_Pa.columns) -2 - k
        ax2.fill_between(pressure_Pa_raw['t(s)'],pressure_Pa_raw['Mean_'+str(k)]+pressure_Pa_raw['Std_'+str(k)],
                                                pressure_Pa_raw['Mean_'+str(k)]-pressure_Pa_raw['Std_'+str(k)],
                                                facecolor=colorsRaw[k], alpha=0.2,label='$P_{Er}$('+str(np.round(colLength-0.2-j*0.4,1))+'m)')

# Set Figure Fonts
ax2.set_ylabel('$P$(Pa)', fontsize=16)
for tick in ax2.xaxis.get_major_ticks():
                tick.label.set_fontsize(14)
for tick in ax2.yaxis.get_major_ticks():
                tick.label.set_fontsize(14) 
ax2.set_xlabel('$t$(s)', fontsize=16)

# Set Limits
ax2.set_xlim([0,tEnd])
ax2.set_ylim([90000,175000])

# Set Legend 
handles, labels = plt.gca().get_legend_handles_labels()
order = legOrder
ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol=legNCol,fontsize= legFontSize)
# plt.show()

# Set Figure Size
fig2.set_size_inches(10, 7)

# Save Figure
fig2.savefig('./Images/'+pressureResFile+'.png',dpi=400)

## Fluid Loss Mass(g)
fig3,ax3 = plt.subplots()
fig4,ax4 = plt.subplots()
ax5=ax3.twinx()
# Fluid Loss Region Properties
HFluidLoss = 0.2                # m
ROut = 9*0.0254/2               # m
AOut = 2*np.pi*ROut*HFluidLoss  # m²
rhoOut = 1000                   # kg/m³

# 
# fluidLossMassDF = pd.concat([passed_seconds,fluidLossMass_g[fluidLossMass]],axis=1)
# fluidLossMassDF.columns=['t(s)','m(g)']
# fluidLossMassDF.drop(fluidLossMassDF[interestData].index, inplace = False)
# fluidLossMassDF['mDot(kg/s)'] = fluidLossMassDF['m(g)'].diff()/fluidLossMassDF['t(s)'].diff()/1000
# fluidLossMassDF.fillna(method = 'bfill',inplace=True)
# fluidLossMassDF['Vr(m/s)']=fluidLossMassDF['mDot(kg/s)']*(1/(rhoOut*AOut)) 

# Calculate Experimental Mean and Standard Deviation
fluidLossMassDF['Mean_m'] = fluidLossMassDF.iloc[:,2:4].mean(axis=1)
fluidLossMassDF['Std_m'] = fluidLossMassDF.iloc[:,2:4].std(axis=1,numeric_only=True)

fluidLossMassPlot = fluidLossMassDF[interestData]
x_data = fluidLossMassPlot['t(s)']
y_data = fluidLossMassPlot['mDot(kg/s)']



# # y_data[y_data <= ]
def fitFunc(t, a, b, c, d, e, ts):
    ti = 500
    # return a*t + b
    # return a*pow(1-b,t)
    y = (t>=ti)*(a/(pow(3*t,b)) - (1/(c))*np.exp((t-ts)/(d)) + e) + (t<ti)*((4e-5/600)*t)
    return y
    # return a/(pow(t,b)) - (1/(c))*np.exp((t-ts)/(d))

def func(x, a, b, c):

    return a * np.exp(-b * x) + c

# # Fit 
a= 5.9 # ^= >
b=1.62 # < = >
c=50000 # ^ = ---^
d=31000
e=2.85e-5
ts=1200
params = [a, b, c, d, e, ts]
# # params, params_covariance = optimize.curve_fit(fitFunc, x_data, y_data, p0=params)
# # params, params_covariance = optimize.curve_fit(func, x_data, y_data, p0=[a,b,c])
# print(params)

# print(fluidLossMassPlot)
ax4.plot(x_data,fluidLossMassPlot.iloc[:,2],  
                        color='none',marker = '^',markeredgecolor = 'blue',
                        label='$m_{FL_{e1}}$')
ax4.plot(x_data,fluidLossMassPlot.iloc[:,3],  
                        color='none',marker = 'o',markeredgecolor = 'blue',
                        label='$m_{FL_{e2}}$')
ax4.legend(fontsize=12)

fluidLossMassPlot.plot(kind='scatter',x='t(s)',y='mDot(kg/s)', ax=ax5, 
                        color='orange',label='$\dot{m}_{FL}$')
ax3.fill_between(x_data,fluidLossMassPlot.iloc[:,5]+fluidLossMassPlot.iloc[:,6],
                        fluidLossMassPlot.iloc[:,5]-fluidLossMassPlot.iloc[:,6],
                        label='$m_{FL_{Er}}$',alpha=0.2)
fluidLossMassPlot.plot(kind='scatter',x='t(s)',y='m(g)', ax=ax3,  
                        color='blue',label='$m_{FL_{Avg}}$')#legend=None)
ax5.plot(x_data, fitFunc(x_data, *params), 'k--',\
        label='$FL_{fit}$')
        # label='fit: a=%5.4g, b=%5.4g, c=%5.4g, d=%5.4g, e=%5.4g, ts=%5.4g' % tuple(params))
# ax3.set_title(testDate)
ax3.set_ylabel('$m_{FL}$(g)',fontsize=14,color='blue')
ax5.set_ylabel('$\dot{m}_{FL}$(kg/s)',fontsize=14,color='orange')
ax3.legend(fontsize=12,loc="upper left")
ax5.legend(fontsize=12,loc="upper right")
ax5.set_xlim([0,tEnd])
ax3.set_xlabel('$t$(s)')
ax3.set_ylim([0,120])
ax4.set_ylabel('$m_{FL}$(g)',fontsize=14)
ax4.set_xlabel('$t$(s)',fontsize=14)
ax4.set_ylim([0,110])
ax4.set_xlim([0,tEnd])
ax5.set_ylim([0,0.00005])
# # plt.show()
fig3.savefig('./Images/'+flowrateResFile2+'.png',dpi=400)
fig4.savefig('./Images/'+flowrateResFile+'.png',dpi=400)