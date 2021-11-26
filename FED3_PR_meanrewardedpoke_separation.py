import pandas as pd
import os
from pandas import read_csv
from pandas import DataFrame

'INPUT'
# enter path to FED file
filename = '/Users/alexreichenbach/Desktop/FiberPhotometry/AgrpCrat_striatum/FED/DSPR/kofed/AgrpCrat2684-200610-113856_zall_activeA.csv'

path = os.path.dirname (filename)

df = pd.read_csv(filename,header=None)


# transpose data to make it easier copying into prism afterwards

df = df.transpose()

# define columns that are rewarded (PR schedule) - you have to #comment out the columns that are not in the input file

rewarded = [0,1,3,7,13,22,34]#,49]#,69,94] 
nonrewarded =[2,4,5,6,8,9,10,11,12,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,35,36,37,38,39,40,41,42,43,44,45,46,47]#,48,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68]#,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93]


df_rewarded = df.iloc[:, rewarded].copy()
df_nonrewarded = df.iloc[:, nonrewarded].copy()
print (df_rewarded)
print (df_nonrewarded)

# average over all non rewarded pokes

df_meannonrewarded = df_nonrewarded.mean (axis =1)
print (df_meannonrewarded)
# create new dataframe starting with rewarded followed by nonrewarded pokes

rewarded = df_rewarded
nonrewarded = df_nonrewarded

df_new_order = pd.concat([rewarded, nonrewarded], axis=1)

#print (df_new_order)
'OUTPUT'
# Save sorted pokes in same directory as "filename.csv"
df_meannonrewarded.to_csv(os.path.join(path,os.path.basename (filename) + "_nonrewardedmean" + ".csv"), index = False)
df_new_order.to_csv(os.path.join(path,os.path.basename (filename) + "_separated" + ".csv"), index = False)
print("Printed new CSV")