# Load the Pandas libraries with alias 'pd' 
import pandas as pd
from matplotlib import pyplot as plt

# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
# df = pd.read_csv(".Labeled/20150429.xlsx") 

# Read Experiment Dataframe
df = pd.read_excel(r'./Labeled/20151005.xlsx', sheet_name='Data')
df['DATETIME'] = pd.to_datetime(df[['YEAR','MONTH','DAY','HOUR','MINUTE','SECOND']])

# Sensor Tag
sensorTag = 'P_6C'
fluidLossMass = 'FL_M_2A'

# psi to Pa Conversion Factor
psi2Pa = 6894.76

# Zone of Interest
df = df.set_index(pd.DatetimeIndex(df['DATETIME']))
df = df[(df['PASSED_SECONDS']<=22500)]
df = df[(df['PASSED_SECONDS']>=3000)]
fig,ax = plt.subplots(1,2)

# Pressure(psi)
presure_psi = df.set_index('DATETIME')[sensorTag]
presure_psi.plot(ax = ax[0],label=sensorTag)

# Resampled Pressure(psi)
filter = (df['MINUTE']%10==0)
groups = df[filter].groupby(['YEAR','MONTH','DAY','HOUR','MINUTE']).first().set_index('DATETIME')
groups[sensorTag].plot(ax=ax[0],label=sensorTag+'_resampled')
print(groups.iloc[0:10])

# Resampled Pressure(g)
groups[fluidLossMass].plot(ax=ax[1], label=fluidLossMass,color='red')

# Show Plot
leg1 = ax[0].legend()
leg2 = ax[1].legend()

plt.show()
a=1
# Define Time Series for each pressure sensor\
# presure_pa_1 = df.set_index('DATETIME')['P_1B']
# presure_pa_2 = df.set_index('DATETIME')['P_2A','P_2B','P_2C']
# presure_pa_3 = df.set_index('DATETIME')['P_2A','P_2B','P_2C']

# # Show Header
# print(presure_pa_1.head(10))