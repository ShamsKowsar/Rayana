from scipy.signal import butter, sosfilt
import warnings
from McsPy import ureg, Q_
import time
import McsPy.McsData
import McsPy
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import os
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

warnings.filterwarnings('ignore')
# adjust this to your local environment
test_data_folder = r'..\Test Data'
file_path = os.path.join(
    test_data_folder, 'MBE1-1st.trial-I14487-25KHz-20220623-1330.h5')
file = McsPy.McsData.RawData(file_path)
electrode_stream = file.recordings[0].analog_streams[0]


def plot_analog_stream_channel2(analog_stream, from_in_s=0, to_in_s=None, show=True):
    global fig
    ids = [c.channel_id for c in analog_stream.channel_infos.values()]
    channel_id = ids[0]
    channel_info = analog_stream.channel_infos[channel_id]
    sampling_frequency = channel_info.sampling_frequency.magnitude
    from_idx = max(0, int(from_in_s * sampling_frequency))
    if to_in_s is None:
        to_idx = analog_stream.channel_data.shape[1]
    else:
        to_idx = min(analog_stream.channel_data.shape[1], int(
            to_in_s * sampling_frequency))

    signal = analog_stream.get_channel_in_range2(channel_id, from_idx, to_idx)
    scale_factor_for_uV = Q_(1, signal[1]).to(ureg.uV).magnitude
    signal_in_uV = signal[0] * scale_factor_for_uV

    return signal_in_uV, sampling_frequency


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    filtered_data = sosfilt(sos, data)
    return filtered_data



def detect_threshold(data, freq):
    # filtered_data = bandpass_filter(data, 300, 3000, int(freq))
    noise_mad = np.median(np.absolute(data)) / 0.6745
    noise_mad
    spike_threshold = -5 * noise_mad
    return spike_threshold


def detect_threshold_crossings(signal, fs, threshold, dead_time):
    dead_time_idx = dead_time * fs
    threshold_crossings = np.diff(
        (signal <= threshold).astype(int) > 0).nonzero()[0]
    distance_sufficient = np.insert(
        np.diff(threshold_crossings) >= dead_time_idx, 0, True)
    while not np.all(distance_sufficient):
        # repeatedly remove all threshold crossings that violate the dead_time
        threshold_crossings = threshold_crossings[distance_sufficient]
        distance_sufficient = np.insert(
            np.diff(threshold_crossings) >= dead_time_idx, 0, True)
    return threshold_crossings


def get_next_minimum(signal, index, max_samples_to_search):
   
    search_end_idx = min(index + max_samples_to_search, len(signal))
    min_idx = np.argmin(signal[index:search_end_idx])
    return index + min_idx


def align_to_minimum(signal, fs, threshold_crossings, search_range):

    search_end = int(search_range*fs)
    aligned_spikes = [get_next_minimum(
        signal, t, search_end) for t in threshold_crossings]
    return np.array(aligned_spikes)


def spike_detection(data, freq):
    signal=data
    spike_threshold = detect_threshold(signal, freq)
    fs = int(electrode_stream.channel_infos[0].sampling_frequency.magnitude)
    pre = 0.001  # 1 ms
    post = 0.002  # 2 ms
    crossings = detect_threshold_crossings(
        signal, fs, spike_threshold, 0.003)  # dead time of 3 ms
    
    spks = align_to_minimum(signal, fs, crossings, 0.002)  # search range 2 ms
    timestamps = spks / fs
    range_in_s = (0, 10)
    spikes_in_range = timestamps[(timestamps >= range_in_s[0]) & (
        timestamps <= range_in_s[1])]
    pre = 0.001  # 1 ms
    post = 0.002  # 2 ms
    c = extract_waveforms(signal, fs, spks, pre, post)
    return c, spikes_in_range


def extract_waveforms(signal, fs, spikes_idx, pre, post):

    c = []
    pre_idx = int(pre * fs)
    post_idx = int(post * fs)
    for index in spikes_idx:
        if index-pre_idx >= 0 and index+post_idx <= len(signal):
            cutout = signal[(index-pre_idx):(index+post_idx)]
            c.append(cutout)
    return [] if c==[] else np.stack(c)


def plot_waveforms(c, fs, pre, post, n=100, color='k', show=True):

    if n is None:
        n = c.shape[0]
    n = min(n, c.shape[0])
    time_in_us = np.arange(-pre*1000, post*1000, 1e3/fs)

    for i in range(n):
        plt.plot(time_in_us, c[i,]*1e6, color, linewidth=1, alpha=0.3)
        plt.xlabel('Time (%s)' % ureg.ms)
        plt.ylabel('Voltage (%s)' % ureg.uV)
        plt.title('c')

