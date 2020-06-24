# Load the Pandas libraries with alias 'pd' 
import pandas as pd
from matplotlib import pyplot as plt

# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
# df = pd.read_csv(".Labeled/20150429.xlsx") 

# Read Experiment Dataframe
df = pd.read_excel(r'./Labeled/20151124.xlsx', sheet_name='Data')
df['DATETIME'] = pd.to_datetime(df[['YEAR','MONTH','DAY','HOUR','MINUTE','SECOND']])


sensorTag = 'P_3B'
presure_psi = df.set_index('DATETIME')[sensorTag]
df = df.set_index(pd.DatetimeIndex(df['DATETIME']))
df = df[(df['PASSED_SECONDS']<=20000)]
filter = (df['MINUTE']%10==0)
fig,ax1 = plt.subplots()
df[sensorTag].plot(ax = ax1)

groups = df[filter].groupby(['YEAR','MONTH','DAY','HOUR','MINUTE']).first().set_index('DATETIME')
groups[sensorTag].plot(ax=ax1)
print(groups.iloc[0:10])
plt.show()
a=1
# Define Time Series for each pressure sensor\
# presure_pa_1 = df.set_index('DATETIME')['P_1B']
# presure_pa_2 = df.set_index('DATETIME')['P_2A','P_2B','P_2C']
# presure_pa_3 = df.set_index('DATETIME')['P_2A','P_2B','P_2C']

# # Show Header
# print(presure_pa_1.head(10))