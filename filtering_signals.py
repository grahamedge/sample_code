import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
# (if you want to use in an iPython notebook)
# %matplotlib inline 


def create_signal():
	'''create a fake signal with which to play'''

	#create a time vector
	t = np.linspace(0, 100, 10000)
	#create a clean signal that has three frequency components
	x = 1.3*np.sin(2*np.pi*1*t) + 0.4*np.sin(2*np.pi*7*t) + 0.6*np.sin(2*np.pi*(0.24)*t) 
	
	#add an offset and linear drift
	x = x + (0.02*t + 2.7)

	#add some random noise to each of the len(t) points in the signal
	# (normally distributed noise with standard deviation = 1 and mean = 0)
	x = x + np.random.normal(0, 1, len(t))

	return x

def bin_data(x, bin_size = 10):
    '''
    take the number of data points that will fit nicely into bins
    	( x[0:binsize*n_bins] ) and then use numpy's .reshape method
    	to turn the 1D vector into a 2D array with shape n_bins x bin_size
    
    THEN average along axis 1 of this array, so that you end up with a
    	n_bins x 1 array for which each entry is the mean of bin_size different
    	data points
    	'''
    
    #determine the number of bins to use
    n_bins = int(np.floor(len(x)/bin_size))

    #bin the data
    x_binned = x[0:bin_size*n_bins].reshape((n_bins,bin_size)).mean(axis=1)

    return x_binned

def low_pass_filter(x, f_samp, tau = 2):
	'''
	use scipy's signal package to apply a low-pass filter to the data 'x'
	which represents equally spaced samples over time with sampling frequency
	'f_samp'

	'tau' is the desired time constant for the low-pass filter

	the filter applied is a 3rd order butterworth filter, but many others
	are available in scipy.signals
	'''
	
	#need to use 'tau' to define the desired cut-off frequency for the
	# filter, which needs to be normalized by the sampling frequency of
	# the data before it can be passed to scipy.signals
	f_cutoff = 1.0/tau
	f_cutoff_norm = f_cutoff / f_samp

	#define the 3rd order butterworth filter
	b, a = signal.butter(3, f_cutoff_norm)

	#apply the filter to the data
	# (the filtfilt method applies the filter once forward in time
	#	and once backward in time to avoid introducing timing delays)
	x_filtered = signal.filtfilt(b, a, x)

	return x_filtered


#generate the signal, and various filtered versions of it, and then plot!

sig = create_signal()

binned = bin_data(sig)

#here I arbitrarily state that the sampling frequency is 10Hz
f_sampl = 10
filtered_A = low_pass_filter(sig, f_sampl, tau = 1) 

filtered_B = low_pass_filter(sig, f_sampl, tau = 10)

t = np.linspace(0, 100, len(sig))
t_binned = np.linspace(0, 100, len(binned))
#plot signal in black
plt.subplot(411)
plt.plot(t, sig, '-k') 

#plot binned signal in red
plt.subplot(412)
plt.plot(t_binned, binned, '--r')
#plot the slightly filtered data in blue
plt.subplot(413)
plt.plot(t, filtered_A, '-b')
#plot the highly filtered data in magenta
plt.subplot(414)
plt.plot(t, filtered_B, '--m')

plt.show()