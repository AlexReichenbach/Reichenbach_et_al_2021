
#import the read_block function from the tdt package
#also import other python packages we care about
from tdt import read_block, download_demo_data, StructType
import numpy as np
import matplotlib.pyplot as plt  # standard Python plotting library


#if you want to learn about all the available options of read_block, uncomment 
#the print() line below:
#print(read_block.__doc__)

#declare a string that will be used as an argument for read_block
#change this to be the full file path to your block files
BLOCKPATH = '/Volumes/FibrePhotometryData/Alex_NAc_PRfast-211116-085056/NAc_3144W_3069W-211119-110819'

#call read block - new variable 'data' is the full data structure
data = read_block(BLOCKPATH)

#what is inside data? print to list out objects of 'data'
print(data)

#where are my demodulated streams? 
#data.streams contains your response traces and the raw photodetector sigs
print(data.streams)

#print out values from my YCRo stream. This is a numpy array for reference, 
#which is a python array that you can do easy math on
#print("data stream _475A")
#print(data.streams._475A.data)

#make some variables up here to so if they change in new recordings you won't
#have to change everything downstream

GCaMP = '_475A'
ISOS = '_415A'
PELLET = 'Bplt'
active = 'Blft'
#inactive = 'Rght'


#print out values from my YCRo stream. This is a numpy array for reference, 
#which is a python array that you can do easy math on
#---------------------------------------------------------------------
# for setupA un-comment these lines
"""
#print("data stream _465A")
#print(data.streams._465A.data)

#make some variables up here to so if they change in new recordings you won't
#have to change everything downstream
GCaMP = '_465A'
ISOS = '_405A'
PELLET = 'Pelt'
active = 'Left'

"""


#same as print(data.streams._475A.data)
#print(data.streams[GCaMP])

#make a time array of our data
num_samples = len(data.streams[GCaMP].data)
time = np.linspace(1, num_samples, num_samples) / data.streams[GCaMP].fs

#plot the demodulated data traces
#this is all matplot lib stuff which you will have to learn
#best way to learn this is to look up examples + stackoverflow
fig1 = plt.subplots(figsize=(10,6))
p1, = plt.plot(time,data.streams[GCaMP].data,color='goldenrod',label='GCaMP')
p2, = plt.plot(time,data.streams[ISOS].data,color='firebrick',label='ISOS')
plt.title('Demodulated Data Traces',fontsize=16)
plt.legend(handles=[p1,p2],loc='lower right',bbox_to_anchor=(1.0,1.01))
plt.autoscale(tight=True)
#plt.show()

#artefact removal


# There is often a large artifact on the onset of LEDs turning on
# Remove data below a set time t
t = 4
inds = np.where(time>t)
ind = inds[0][0]
time = time[ind:] # go from ind to final index
data.streams[GCaMP].data = data.streams[GCaMP].data[ind:]
data.streams[ISOS].data = data.streams[ISOS].data[ind:]

# Plot again at new time range
fig2 = plt.figure(figsize=(10, 6))
ax1 = fig2.add_subplot(111)

# Plotting the traces
p1, = ax1.plot(time,data.streams[GCaMP].data, linewidth=2, color='green', label='GCaMP')
p2, = ax1.plot(time,data.streams[ISOS].data, linewidth=2, color='blueviolet', label='ISOS')

ax1.set_ylabel('mV')
ax1.set_xlabel('Seconds')
ax1.set_title('Raw Demodulated Responsed with Artifact Removed')
ax1.legend(handles=[p1,p2],loc='upper right')
fig2.tight_layout()
# fig


#downsampling data and local averaging


# Average around every Nth point and downsample Nx
N = 10 # Average every 10 samples into 1 value
F415 = []
F475 = []

for i in range(0, len(data.streams[GCaMP].data), N):
    F475.append(np.mean(data.streams[GCaMP].data[i:i+N-1])) # This is the moving window mean
data.streams[GCaMP].data = F475

for i in range(0, len(data.streams[ISOS].data), N):
    F415.append(np.mean(data.streams[ISOS].data[i:i+N-1]))
data.streams[ISOS].data = F415

#decimate time array to match length of demodulated stream
time = time[::N] # go from beginning to end of array in steps on N
time = time[:len(data.streams[GCaMP].data)]

# Detrending and dFF
# Full trace dFF according to Lerner et al. 2015
# http://dx.doi.org/10.1016/j.cell.2015.07.014
# dFF using 405 fit as baseline

x = np.array(data.streams[ISOS].data)
y = np.array(data.streams[GCaMP].data)
bls = np.polyfit(x, y, 1)
Y_fit_all = np.multiply(bls[0], x) + bls[1]
Y_dF_all = y - Y_fit_all

dFF = np.multiply(100, np.divide(Y_dF_all, Y_fit_all))
std_dFF = np.std(dFF)


# organising nose pokes and pellets during recording


# First make a continous time series of TTL events (epocs) and plot
PELLET_ON = data.epocs[PELLET].onset
PELLET_OFF = data.epocs[PELLET].offset
# Add the first and last time stamps to make tails on the TTL stream
PELLET_x = np.append(np.append(time[0], np.reshape(np.kron([PELLET_ON, PELLET_OFF],
                   np.array([[1], [1]])).T, [1,-1])[0]), time[-1])
sz = len(PELLET_ON)
d = data.epocs[PELLET].data
# Add zeros to beginning and end of 0,1 value array to match len of PELLET_x
PELLET_y = np.append(np.append(0,np.reshape(np.vstack([np.zeros(sz),
    d, d, np.zeros(sz)]).T, [1, -1])[0]),0)

y_scale = 15 #adjust according to data needs
y_shift = -6 #scale and shift are just for asthetics


# First subplot in a series: dFF with lick epocs
fig3 = plt.figure(figsize=(20,12))
ax2 = fig3.add_subplot(311)

p1, = ax2.plot(time, dFF, linewidth=2, color='green', label='GCaMP')
p2, = ax2.plot(PELLET_x, y_scale*PELLET_y+y_shift, linewidth=2, color='dodgerblue', label='Pellet')
#p3, = ax2.plot(active_x, y_scale*active_y+y_shift, linewidth=2, color='red', label='Poke')
ax2.set_ylabel(r'$\Delta$F/F')
ax2.set_xlabel('Seconds')
ax2.set_title('dopamine response')
ax2.legend(handles=[p1,p2], loc='upper left')
fig3.tight_layout()
#plt.show()

# # Lick Bout Logic
# Now combine lick epocs that happen in close succession to make a single on/off event (a lick BOUT). Top view logic: if difference between consecutive lick onsets is below a certain time threshold and there was more than one lick in a row, then consider it as one bout, otherwise it is its own bout. Also, make sure a minimum number of licks was reached to call it a bout.

PELLET_EVENT = 'PELLET_EVENT'

PELLET_DICT = {
        "name":PELLET_EVENT,
        "onset":[],
        "offset":[],
        "type_str":data.epocs[PELLET].type_str,
        "data":[]
        }

print(PELLET_DICT)
#pass StructType our new dictionary to make keys and values
data.epocs.PELLET_EVENT = StructType(PELLET_DICT)

pellet_on_diff = np.diff(data.epocs[PELLET].onset)
BOUT_TIME_THRESHOLD = 1
pellet_diff_ind = np.where(pellet_on_diff >= BOUT_TIME_THRESHOLD)[0]
#for some reason np.where returns a 2D array, hence the [0]

# Make an onset/ offset array based on threshold indicies
diff_ind = 0
for ind in pellet_diff_ind: 
    # BOUT onset is thresholded onset index of lick epoc event
    data.epocs[PELLET_EVENT].onset.append(data.epocs[PELLET].onset[diff_ind])
    # BOUT offset is thresholded offset of lick event before next onset
    data.epocs[PELLET_EVENT].offset.append(data.epocs[PELLET].offset[ind])
    # set the values for data, arbitrary 1
    data.epocs[PELLET_EVENT].data.append(1)
    diff_ind = ind + 1

# special case for last event to handle lick event offset indexing
data.epocs[PELLET_EVENT].onset.append(data.epocs[PELLET].onset[pellet_diff_ind[-1]+1])
data.epocs[PELLET_EVENT].offset.append(data.epocs[PELLET].offset[-1])
data.epocs[PELLET_EVENT].data.append(1)

# Now determine if it was a 'real' lick bout by thresholding by some
# user-set number of licks in a row
MIN_PELLET_THRESH = 1 #four licks or more make a bout
pellet_array = []

# Find number of licks in pellet_array between onset and offset of 
# our new lick BOUT PELLET_EVENT
for on, off in zip(data.epocs[PELLET_EVENT].onset,data.epocs[PELLET_EVENT].offset):
    pellet_array.append(
        len(np.where((data.epocs[PELLET].onset >= on) & (data.epocs[PELLET].onset <= off))[0]))

# Remove onsets, offsets, and data of thrown out events
pellet_array = np.array(pellet_array)
inds = np.where(pellet_array<MIN_PELLET_THRESH)[0]
for index in sorted(inds, reverse=True):
    del data.epocs[PELLET_EVENT].onset[index]
    del data.epocs[PELLET_EVENT].offset[index]
    del data.epocs[PELLET_EVENT].data[index]
    
# Make a continuous time series for lick BOUTS for plotting
PELLET_EVENT_on = data.epocs[PELLET_EVENT].onset
PELLET_EVENT_off = data.epocs[PELLET_EVENT].offset
PELLET_EVENT_x = np.append(time[0], np.append(
    np.reshape(np.kron([PELLET_EVENT_on, PELLET_EVENT_off],np.array([[1], [1]])).T, [1,-1])[0], time[-1]))
sz = len(PELLET_EVENT_on)
d = data.epocs[PELLET_EVENT].data
PELLET_EVENT_y = np.append(np.append(
    0, np.reshape(np.vstack([np.zeros(sz), d, d, np.zeros(sz)]).T, [1 ,-1])[0]), 0)

#--------------------------
#active = 'Left'
#inactive = 'Rght'


# First make a continous time series of active TTL events (epocs) and plot
active_on = data.epocs[active].onset
active_off = data.epocs[active].offset
# Add the first and last time stamps to make tails on the TTL stream
active_x = np.append(np.append(time[0], np.reshape(np.kron([active_on, active_off],
                   np.array([[1], [1]])).T, [1,-1])[0]), time[-1])
sz = len(active_on)
d = data.epocs[active].data
# Add zeros to beginning and end of 0,1 value array to match len of active_x
active_y = np.append(np.append(0,np.reshape(np.vstack([np.zeros(sz),
    d, d, np.zeros(sz)]).T, [1, -1])[0]),0)

# # Lick Bout Logic
# Now combine lick epocs that happen in close succession to make a single on/off event (a lick BOUT). Top view logic: if difference between consecutive lick onsets is below a certain time threshold and there was more than one lick in a row, then consider it as one bout, otherwise it is its own bout. Also, make sure a minimum number of licks was reached to call it a bout.

active_EVENT = 'active_EVENT'

active_DICT = {
        "name":active_EVENT,
        "onset":[],
        "offset":[],
        "type_str":data.epocs[active].type_str,
        "data":[]
        }

print(active_DICT)
#pass StructType our new dictionary to make keys and values
data.epocs.active_EVENT = StructType(active_DICT)

active_on_diff = np.diff(data.epocs[active].onset)
BOUT_TIME_THRESHOLD = 1
active_diff_ind = np.where(active_on_diff >= BOUT_TIME_THRESHOLD)[0]
#for some reason np.where returns a 2D array, hence the [0]

# Make an onset/ offset array based on threshold indicies
diff_ind = 0
for ind in active_diff_ind: 
    # BOUT onset is thresholded onset index of lick epoc event
    data.epocs[active_EVENT].onset.append(data.epocs[active].onset[diff_ind])
    # BOUT offset is thresholded offset of lick event before next onset
    data.epocs[active_EVENT].offset.append(data.epocs[active].offset[ind])
    # set the values for data, arbitrary 1
    data.epocs[active_EVENT].data.append(1)
    diff_ind = ind + 1

# special case for last event to handle lick event offset indexing
data.epocs[active_EVENT].onset.append(data.epocs[active].onset[active_diff_ind[-1]+1])
data.epocs[active_EVENT].offset.append(data.epocs[active].offset[-1])
data.epocs[active_EVENT].data.append(1)

# Now determine if it was a 'real' lick bout by thresholding by some
# user-set number of licks in a row
MIN_active_THRESH = 1 #four licks or more make a bout
actives_array = []

# Find number of licks in pellet_array between onset and offset of 
# our new lick BOUT PELLET_EVENT
for on, off in zip(data.epocs[active_EVENT].onset,data.epocs[active_EVENT].offset):
    actives_array.append(
        len(np.where((data.epocs[active].onset >= on) & (data.epocs[active].onset <= off))[0]))

# Remove onsets, offsets, and data of thrown out events
actives_array = np.array(actives_array)
inds = np.where(actives_array<MIN_active_THRESH)[0]
for index in sorted(inds, reverse=True):
    del data.epocs[active_EVENT].onset[index]
    del data.epocs[active_EVENT].offset[index]
    del data.epocs[active_EVENT].data[index]
    
# Make a continuous time series for lick BOUTS for plotting
active_EVENT_on = data.epocs[active_EVENT].onset
active_EVENT_off = data.epocs[active_EVENT].offset
active_EVENT_x = np.append(time[0], np.append(
    np.reshape(np.kron([active_EVENT_on, active_EVENT_off],np.array([[1], [1]])).T, [1,-1])[0], time[-1]))
sz = len(active_EVENT_on)
d = data.epocs[active_EVENT].data
active_EVENT_y = np.append(np.append(
    0, np.reshape(np.vstack([np.zeros(sz), d, d, np.zeros(sz)]).T, [1 ,-1])[0]), 0)
#-------------------------
# # Plot dFF with newly defined lick bouts


ax3 = fig3.add_subplot(312)
p1, = ax3.plot(time, dFF, linewidth=2, color='green', label='465 nm')
p2, = ax3.plot(PELLET_EVENT_x, y_scale*PELLET_EVENT_y+y_shift, linewidth=2, color='magenta', label='Pellet')
p3, = ax3.plot(active_EVENT_x, y_scale*active_EVENT_y+y_shift, linewidth=1, color='blue', label='Poke')
ax3.set_ylabel(r'$\Delta$F/F')
ax3.set_xlabel('Seconds')
ax3.set_title('PR session')
ax3.legend(handles=[p1, p2, p3], loc='upper left')
fig3.tight_layout()
fig3
#plt.show()

# # Make nice area fills instead of epocs for asthetics

ax4 = fig3.add_subplot(313)
p1, = ax4.plot(time, dFF,linewidth=2, color='green', label='dopamine')
for on, off in zip(data.epocs[PELLET_EVENT].onset, data.epocs[PELLET_EVENT].offset):
    ax4.axvspan(on, off, alpha=0.25, color='dodgerblue')
for on, off in zip(data.epocs[active_EVENT].onset, data.epocs[active_EVENT].offset):
    ax4.axvspan(on, off, alpha=0.25, color='red')    
ax4.set_ylabel(r'$\Delta$F/F')
ax4.set_xlabel('Seconds')
ax4.set_title(' ')
fig3.tight_layout()
fig3
#plt.show()

# # Time Filter Around Pellet Epocs
# Note that we are using dFF of the full time series, not peri-event dFF where f0 is taken from a pre-event basaeline period.


PRE_TIME = 5 # five seconds before event onset
POST_TIME = 10 # ten seconds after
fs = data.streams[GCaMP].fs/N #recall we downsampled by N = 10 earlier

# time span for peri-event filtering, PRE and POST, in samples
TRANGE = [-PRE_TIME*np.floor(fs), POST_TIME*np.floor(fs)]

dFF_snips = []
array_ind = []
pre_stim = []
post_stim = []

for on in data.epocs[PELLET_EVENT].onset:
    # If the bout cannot include pre-time seconds before event, make zero
    if on < PRE_TIME:
        dFF_snips.append(np.zeros(TRANGE[1]-TRANGE[0]))
    else: 
        # find first time index after bout onset
        array_ind.append(np.where(time > on)[0][0])
        # find index corresponding to pre and post stim durations
        pre_stim.append(array_ind[-1] + TRANGE[0])
        post_stim.append(array_ind[-1] + TRANGE[1])
        dFF_snips.append(dFF[int(pre_stim[-1]):int(post_stim[-1])])
        
# Make all snippets the same size based on min snippet length
min1 = np.min([np.size(x) for x in dFF_snips])
dFF_snips = [x[1:min1] for x in dFF_snips]

mean_dFF_snips = np.mean(dFF_snips, axis=0)
std_dFF_snips = np.std(mean_dFF_snips, axis=0)

peri_time = np.linspace(1, len(mean_dFF_snips), len(mean_dFF_snips))/fs - PRE_TIME


# # Make a Peri-Event Stimulus Plot and Heat Map

fig4 = plt.figure(figsize=(6,10))
ax5 = fig4.add_subplot(211)

for snip in dFF_snips:
    p1, = ax5.plot(peri_time, snip, linewidth=.5, color=[.7, .7, .7], label='Individual Trials')
p2, = ax5.plot(peri_time, mean_dFF_snips, linewidth=2, color='green', label='Mean Response')

# Plotting standard error bands
p3 = ax5.fill_between(peri_time, mean_dFF_snips+std_dFF_snips, 
                      mean_dFF_snips-std_dFF_snips, facecolor='green', alpha=0.2)
p4 = ax5.axvline(x=0, linewidth=3, color='slategray', label='Lick Bout Onset')

ax5.axis('tight')
ax5.set_xlabel('Seconds')
ax5.set_ylabel(r'$\Delta$F/F')
ax5.set_title('Peri-Event Pellet retrieval')
ax5.legend(handles=[p1, p2, p4], bbox_to_anchor=(1.1, 1.05))

ax6 = fig4.add_subplot(212)
cs = ax6.imshow(dFF_snips, cmap=plt.cm.Greys,
                interpolation='none', extent=[-PRE_TIME,POST_TIME,len(dFF_snips),0],)
ax6.set_ylabel('Trial Number')
ax6.set_yticks(np.arange(.5, len(dFF_snips), 2))
ax6.set_yticklabels(np.arange(0, len(dFF_snips), 2))
fig4.colorbar(cs)
fig4

plt.show()


#save list of timing for pellet and nose pokes
import os
import numpy as np
import pylab
import itertools, collections
import matplotlib.pyplot as plt
import pandas as pd


'OUTPUT'
filename = BLOCKPATH

save_to_csv_Pellettiming = False
    # If true then input the file name
filename_Pellettiming = os.path.join (
    #os.path.dirname (filename), # saves file in same directory
    '/Users/alexreichenbach/Desktop/post_doc/AgRP_Crat/Fiphodata/AgrpCrat_fiphoFED_timing/NAc/fast', # change the path to where you want files saved
    os.path.basename(filename) + "_Pellettiming_A" + ".csv")


save_to_csv_Poketiming = False
    # If true then input the file name
filename_Poketiming = os.path.join (
    #os.path.dirname (filename), # saves file in same directory
    '/Users/alexreichenbach/Desktop/post_doc/AgRP_Crat/Fiphodata/AgrpCrat_fiphoFED_timing/NAc/fast', # change the path to where you want files saved
    os.path.basename(filename) + "_Poketiming_A" + ".csv")

if save_to_csv_Pellettiming:
    np.savetxt(filename_Pellettiming, PELLET_ON, delimiter=",")
    print("Printed Pellettiming CSV")

if save_to_csv_Poketiming:
    np.savetxt(filename_Poketiming, active_on, delimiter=",")
    print("Printed Poketiming CSV")