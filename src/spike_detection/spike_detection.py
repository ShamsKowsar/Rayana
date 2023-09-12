import time

import ModifiedMcsPy
import ModifiedMcsPy.McsData
import numpy as np
from ModifiedMcsPy import Q_, ureg
from scipy import signal
from scipy.signal import bessel, butter, lfilter, sosfilt, sosfiltfilt

from functions_and_filters import *


file_path = r'..\..\Test Data\MBE1-1st.trial-I14487-25KHz-20220623-1330.h5'
file = ModifiedMcsPy.McsData.RawData(file_path)
electrode_stream = file.recordings[0].analog_streams[0]


sampling_frequency = get_frequency(electrode_stream)


nyquist_freq = 0.5 * sampling_frequency
normalized_cutoff_freq = 100 / nyquist_freq
sos_1 = bessel(2, normalized_cutoff_freq, btype='highpass',
               output='sos', analog=False, fs=sampling_frequency)


sos_2 = bessel(1, 1, btype='low', output='sos', fs=sampling_frequency)


t = 0
# start=time.time()
while True:
    data = get_data(
        electrode_stream, sampling_frequency, from_in_s=t, to_in_s=t+0.01)
    t = t+0.01
    data = np.round(data, 5)
    filtered_data = filter_data(
        data, 1, sampling_frequency/2-1, sampling_frequency)
    step_1_result = sosfilt(sos_1, filtered_data)
    step_2_result = sosfiltfilt(sos_2, np.abs(step_1_result))
    for i in range(len(step_1_result)):
        spikes=spike_detection(
            step_1_result[i], step_2_result[i], sampling_frequency)

    # if (int(t*100) % 500 == 0):
    #     print("%.6f - %.6f" % (t, time.time()-start))
    #     print('-')
