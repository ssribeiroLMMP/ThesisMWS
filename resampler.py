# Load the Pandas libraries with alias 'pd' 
import pandas as pd
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
filename = '20160114'
Analysis = 'LongCure'
startTime = 3000
endTime = 50000
df = pd.read_excel(r'./Labeled/'+filename+'.xlsx', sheet_name='Data')
df['DATETIME'] = pd.to_datetime(df[['YEAR','MONTH','DAY','HOUR','MINUTE','SECOND']])

# Resample
filterSec = (df['SECOND']>=0)
period = (df['PASSED_SECONDS']>=startTime) & (df['PASSED_SECONDS']<=endTime)
groups = df[filterSec & period].groupby(['YEAR','MONTH','DAY','HOUR','MINUTE']).first().set_index('DATETIME')
final = groups.resample('2T').first()
# Write Resampled File
# groups.to_csv(+filename, mode='a',header=True)
write_df_csv('./Resampled/', filename+'_'+Analysis+'.csv',final)