import json
import socket
import pickle
from McsPy import ureg, Q_
import McsPy
import McsPy.McsData
from scipy import signal
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import signal
import McsPy
import McsPy.McsData
import time
from McsPy import ureg, Q_
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from scipy.signal import butter, sosfilt

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    filtered_data = sosfilt(sos, data)
    return filtered_data

# Example usage
fs = 10000 # sampling rate of the signal
lowcut = 200 # lower cutoff frequency
highcut = 3500 # upper cutoff frequency


def fft_denoiser(x, n_components, to_real=True):
    n = len(x[0])
    fft = np.fft.fft(x)
    c=time.time()
    PSD = fft * np.conj(fft) / n
    c=time.time()
    _mask = PSD > n_components
    fft = _mask * fft
    clean_data = np.fft.ifft(fft)
    c=time.time()
    if to_real:
        clean_data = clean_data.real
    return clean_data


def filter_data(data, low, high, sf, order=2):
    c=time.time()

    data=fft_denoiser(data,200,True)
    c=time.time()
    nyq = sf/2
    low = low/nyq
    high = high/nyq
    c=time.time()
    b, a = butter(order, [low, high], btype='band')
    c=time.time()
    filtered_data = lfilter(b, a, data)
    c=time.time()
    return filtered_data

def plot_analog_stream_channel2(analog_stream, channel_idx, from_in_s=0, to_in_s=None, show=True):
    global fig
    ids = [c.channel_id for c in analog_stream.channel_infos.values()]
    channel_id = ids[channel_idx]
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


# adjust this to your local files

file_path = r'..\Test Data\MBE1-2end.trial-I14487-25KHz-20220623-1350.h5'
print(file_path)
file = McsPy.McsData.RawData(file_path)
electrode_stream = file.recordings[0].analog_streams[0]


HOST = 'localhost'
PORT = 50007
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()


while 1:
    conn, addr = s.accept()
    chunk = int(conn.recv(1024).decode())
    m, n = plot_analog_stream_channel2(
        electrode_stream, 0, from_in_s=chunk, to_in_s=chunk+1)
    arr = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
    data = []
    print(len(m[0]))
    c=time.time()
    for i in range(60):
      filtered_data = bandpass_filter(m[i], lowcut, highcut, fs)
      data.append(filtered_data)
    downsampled_data = downsample_data(np.array(data), int(n), 1000)
    data=((data))
    print(time.time()-c)
    print(len(data[0]))
    data = json.dumps({'m': downsampled_data[0].tolist(),'f':len(downsampled_data[0][0])})

    conn.send(data.encode())

conn.close()
