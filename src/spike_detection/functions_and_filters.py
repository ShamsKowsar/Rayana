import time

import ModifiedMcsPy
import ModifiedMcsPy.McsData
import numpy as np
from ModifiedMcsPy import Q_, ureg
from scipy import signal
from scipy.signal import bessel, butter, lfilter, sosfilt, sosfiltfilt


def detect_threshold_crossings(signal, fs, threshold, dead_time):
    dead_time_idx = dead_time * fs
    threshold_crossings = np.diff(
        (signal <= threshold).astype(int) > 0).nonzero()[0]
    distance_sufficient = np.insert(
        np.diff(threshold_crossings) >= dead_time_idx, 0, True)
    while not np.all(distance_sufficient):
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


def extract_waveforms(signal, fs, spikes_idx, pre, post):

    c = []
    pre_idx = int(pre * fs)
    post_idx = int(post * fs)
    for index in spikes_idx:
        if index-pre_idx >= 0 and index+post_idx <= len(signal):
            cutout = signal[(index-pre_idx):(index+post_idx)]
            c.append(cutout)
    return [] if c == [] else np.stack(c)


def spike_detection(stream_1, stream_2, sampling_freq):
    noise_mad = np.median(np.absolute(stream_2)) / 0.6745
    spike_threshold = -5 * noise_mad
    if not np.any(stream_1 <= spike_threshold):
        return []

    else:
        signal = stream_1
        fs = sampling_freq
        pre = 0.001
        post = 0.002
        crossings = detect_threshold_crossings(
            signal, fs, spike_threshold, 0.002)
        spks = align_to_minimum(signal, fs, crossings, 0.002)
        
        pre = 0.001
        post = 0.002
        c = extract_waveforms(signal, fs, spks, pre, post)
        return spks


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    filtered_data = sosfilt(sos, data)
    return filtered_data


def fft_denoiser(x, n_components, to_real=True):
    n = len(x[0])
    fft = np.fft.fft(x)
    PSD = fft * np.conj(fft) / n
    _mask = PSD > n_components
    fft = _mask * fft
    clean_data = np.fft.ifft(fft)
    if to_real:
        clean_data = clean_data.real
    return clean_data


def filter_data(data, low, high, sf, order=2):

    data = fft_denoiser(data, 200, True)

    nyq = sf/2
    low = low/nyq
    high = high/nyq

    b, a = butter(order, [low, high], btype='band')

    filtered_data = lfilter(b, a, data)

    return filtered_data


def downsample_data(data, sf, target_sf):
    factor = sf/target_sf
    if factor <= 10:
        data_down = signal.decimate(data, int(factor))
    else:
        factor = 10
        data_down = data
        while factor > 1:
            data_down = signal.decimate(data_down, int(factor))
            sf = sf/factor
            factor = int(min([10, sf/target_sf]))
    return data_down, sf

def get_channel_in_range2(self, idx_start=0, idx_end=None):

    channel_id=list(self.channel_infos)[0]
    if idx_start < 0:
        idx_start = 0
    if idx_end is None or idx_end > self.channel_data.shape[1]:
        idx_end = self.channel_data.shape[1]
    else:
        idx_end += 1
    signal = self.channel_data[:, idx_start : idx_end]
    # print(signal.shape)
    scale = self.channel_infos[channel_id].adc_step.magnitude
    #scale = self.channel_infos[channel_id].get_field('ConversionFactor') * (10**self.channel_infos[channel_id].get_field('Exponent'))
    signal_corrected = (signal - self.channel_infos[channel_id].get_field('ADZero'))  * scale
    return (signal_corrected, self.channel_infos[channel_id].adc_step.units)

def get_data(analog_stream,sampling_frequency, from_in_s=0, to_in_s=None, show=True):
    from_idx = max(0, int(from_in_s * sampling_frequency))
    if to_in_s is None:
        to_idx = analog_stream.channel_data.shape[1]
    else:
        to_idx = min(analog_stream.channel_data.shape[1], int(
            to_in_s * sampling_frequency))
    signal = analog_stream.get_channel_in_range2(from_idx, to_idx)
    scale_factor_for_uV = Q_(1, signal[1]).to(ureg.uV).magnitude
    signal_in_uV = signal[0] * scale_factor_for_uV
    return signal_in_uV


def get_frequency(analog_stream):
    ids = [c.channel_id for c in analog_stream.channel_infos.values()]
    channel_id = ids[0]
    channel_info = analog_stream.channel_infos[channel_id]
    sampling_frequency = channel_info.sampling_frequency.magnitude
    return sampling_frequency