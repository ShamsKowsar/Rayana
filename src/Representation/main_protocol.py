
epoch_id=1
chip_id='old_culture_test'
session_type=0
stim_type='coupling'
high_freq_coupling=100.0
low_freq_coupling=5.0
between_stims_rest=10.0
num_cycles=8
num_trials_in_each_cycle=4
between_cycles_rest=10*60
between_trials_rest=20.0
positive_amplitude=15.0
negative_amplitude=-15.0
between_phase_offset=25.0
offset_between_high_and_low_of_biphasic=between_phase_offset/1_000_000

from System import Int32, Int64, Double, String, Single
from matplotlib.colorbar import Colorbar
from queue import Queue
from scipy.signal import butter, lfilter
from threading import Thread, Event, Lock
import System
import clr
import h5py  
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import queue
import random

import scipy.io
import time

start_acq=0
program_start = time.time()

FPS = 10  


game_state_lock = Lock()
stop_event = Event()











base_path = r"C:\Users\BioCAM User\Downloads\API\BioCamDriverAPI_v2.5\BioCamDriverAPI_v2.5\API"
clr.AddReference(os.path.join(base_path, "3Brain.BioCamDriver.dll"))
clr.AddReference(os.path.join(base_path, "3Brain.Common.dll"))




assembly_biocam = System.Reflection.Assembly.LoadFile(os.path.join(base_path, "3Brain.BioCamDriver.dll"))
assembly_common = System.Reflection.Assembly.LoadFile(os.path.join(base_path, "3Brain.Common.dll"))

bioCamPoolType = assembly_biocam.GetType("_3Brain.BioCamDriver.BioCamPool")
dataPacketReceivedEventArgsType = assembly_biocam.GetType("_3Brain.BioCamDriver.DataPacketReceivedEventArgs")
chCoordType = assembly_common.GetType("_3Brain.Common.ChCoord")
rectangularStimPulseType = assembly_common.GetType("_3Brain.Common.RectangularStimPulse")
StimPropertiesType = assembly_common.GetType("_3Brain.Common.StimProperties")
DataSamplingTimeConverterType = assembly_common.GetType("_3Brain.Common.DataSamplingTimeConverter")
StimTrainProtocolType = assembly_biocam.GetType("_3Brain.BioCamDriver.StimTrainProtocol")
StimEndPointDupleXType = assembly_biocam.GetType("_3Brain.BioCamDriver.StimEndPointDupleX")
LoadingProtocolProgressChangedEventArgsType = assembly_biocam.GetType("_3Brain.BioCamDriver.LoadingProtocolProgressChangedEventArgs")



bioCam = None
protocol_manager = None
is_streaming = False
meaPlatePilot = None


spike_counts = None
slot_index = None
latest_timestamp_us = 0  
result = [None,None]
DEBUG = False



default_stim_properties_field = StimPropertiesType.GetField("Default")


default_stim_properties = default_stim_properties_field.GetValue(None)


time_resolution_property = StimPropertiesType.GetProperty("TimeResolutionMicroSec")
time_resolution_value = time_resolution_property.GetValue(default_stim_properties)


amplitude_resolution_property = StimPropertiesType.GetProperty("AmplitudeResolution")
amplitude_resolution_value = amplitude_resolution_property.GetValue(default_stim_properties)


def initialize_biocam():
    global bioCam, slot_index
    try:
        activate_method = bioCamPoolType.GetMethod("Activate")
        activate_method.Invoke(None, [False])
        time.sleep(1)
        
        n_biocams_property = bioCamPoolType.GetProperty("NBioCams")
        n_biocams = n_biocams_property.GetValue(None)
        print(n_biocams)
        if n_biocams == 0:
            print("No BioCAM devices connected. Exiting...")
            return

        take_control_method = bioCamPoolType.GetMethod("TakeFirstFreeBioCamControl")
        bioCam = take_control_method.Invoke(None, [])
        if bioCam is None:
            raise Exception("No available BioCAM devices found!")
            
        
        
        get_slot_indexes_method = bioCamPoolType.GetMethod("GetSlotIndexesConnectedBioCam")
        connected_indexes = get_slot_indexes_method.Invoke(None, [])
        python_indexes = list(connected_indexes)  
        if len(python_indexes) == 0:
            print("Failed to retrieve connected BioCam slot indexes. Exiting...")
            return
        slot_index = python_indexes[0]
        slot_index = Int32(slot_index)  
        print(f"BioCam device is in slot index: {slot_index}")
        
        print("BioCam successfully initialized.")
    except Exception as e:
        print(f"Error during BioCam initialization: {e}")
        bioCam = None
       


def set_chamber_temperature(target_temperature_celsius):
    global bioCam, slot_index, meaPlatePilot
    if bioCam is None:
        print("Error: BioCam not initialized. Please initialize BioCam first.")
        return
    
    try:
       
        
        if meaPlatePilot is None:
            print("Error: No IMeaPlatePilot found!")
            return
        
        
        settings_property = meaPlatePilot.GetType().GetProperty("Settings")
        plate_settings = settings_property.GetValue(meaPlatePilot)
        
        if plate_settings is None:
            print("Error: No MeaPlateSettings found!")
            return
        
        
        
        
        is_temp_control_on_property = plate_settings.GetType().GetProperty("IsChamberTemperatureControlOn")
        is_temp_control_on = is_temp_control_on_property.GetValue(plate_settings)

        print(f"Chamber temperature control is {'ON' if is_temp_control_on else 'OFF'}.")

        
        if not is_temp_control_on:
             is_temp_control_on_property.SetValue(plate_settings, True)
             print("Chamber temperature control has been turned ON.")

        
        set_temperature_property = plate_settings.GetType().GetProperty("SetChamberTemperatureCelsius")
        set_temperature_property.SetValue(plate_settings, Single(target_temperature_celsius))
        
        print(f"Chamber temperature set to {target_temperature_celsius}°C.")
        
        
        while True:
            
            read_temperature_property = plate_settings.GetType().GetProperty("ReadChamberTemperatureCelsius")
            current_temperatures = read_temperature_property.GetValue(plate_settings)
            
            
            if current_temperatures is None or len(current_temperatures) == 0:
                print("Error: No temperature readings available!")
                return
            
            
            current_temperature = current_temperatures[0]
            print(f"Current chamber temperature: {current_temperature}°C")
            
            
            if current_temperature >= target_temperature_celsius:
                print(f"Target temperature of {target_temperature_celsius}°C reached.")
                break  
            
            
            time.sleep(10)  

    except Exception as e:
        print(f"Error setting chamber temperature: {e}")


def set_stimulation_calibration(stimulatorSettings, chip_calibration_delay = 5000):
    
    try:
    
        is_stim_calibration_on = stimulatorSettings.GetType().GetProperty("IsStimCalibrationOn").GetValue(stimulatorSettings)
        if not is_stim_calibration_on:
            is_stim_calibration_on = stimulatorSettings.GetType().GetProperty("IsStimCalibrationOn").SetValue(stimulatorSettings,True)
            print(f"is stimulation calibration on? {is_stim_calibration_on} ")
            
        stimulatorSettings.GetType().GetProperty("CalibrationDistanceMicroSec").SetValue(stimulatorSettings,Single(chip_calibration_delay))

    except Exception as e:    
        print(f"Error setting stimulation callibration: {e}")


def terminate_acquisition():
    global bioCam, is_streaming, slot_index, meaPlatePilot

    if bioCam is None:
        print("No active BioCam to terminate.")
        return

    try:
        
        if is_streaming:
            stop_streaming_method = bioCam.GetType().GetMethod("StopDataStreaming")
            stop_streaming_method.Invoke(bioCam, [])
            is_streaming = False
            print("Data streaming stopped successfully.")

        
        
        
        settings_property = meaPlatePilot.GetType().GetProperty("Settings")
        plate_settings = settings_property.GetValue(meaPlatePilot)
        
        
        is_temp_control_on_property = plate_settings.GetType().GetProperty("IsChamberTemperatureControlOn")
        is_temp_control_on = is_temp_control_on_property.GetValue(plate_settings)

        print(f"Chamber temperature control is {'ON' if is_temp_control_on else 'OFF'}.")

        
        if is_temp_control_on:
             is_temp_control_on_property.SetValue(plate_settings, False)
             print("Chamber temperature control has been turned OFF.")

        
        close_method = bioCam.GetType().GetMethod("Close")
        close_method.Invoke(bioCam, [True])  
        print("BioCam connection closed successfully.")

        
        from System import Int32
        slot_index_int32 = Int32(slot_index)  
        release_control_method = bioCamPoolType.GetMethod("ReleaseBioCamControl")
        release_control_method.Invoke(None, [slot_index_int32])
        print(f"BioCam control released successfully at slot index {slot_index_int32}.")

    except Exception as ex:
        print(f"Error during termination: {ex}")

    finally:
        
        
        bioCam = None
        is_streaming = False
        print("All resources have been reset.")

        print("Acquisition terminated safely.")

def save_streamed_data(h5_filename, data_queue):
    
    global stop_event
    with h5py.File(h5_filename, "w") as h5file:
        
        dset_data = h5file.create_dataset("electrode_data", 
                                          shape=(0,),  
                                          maxshape=(None,),  
                                          dtype='uint16')
        dset_timestamps = h5file.create_dataset("timestamps", 
                                                shape=(0,),  
                                                maxshape=(None,),  
                                                dtype='int64')

        

        buffer_data = []
        buffer_timestamps = []
        expected_samples = None  

        
        batch_size = 10  

        while not stop_event.is_set():
            try:
                
                data, timestamp = data_queue.get(timeout=1)

                
                if data.shape[0] == 0:  
                    continue

                
                if expected_samples is None:
                    expected_samples = data.shape[0]  
                    

                
                if data.shape[0] != expected_samples:
                    
                    continue

                
                buffer_data.append(data)  
                buffer_timestamps.append(timestamp)

                
                if len(buffer_data) >= batch_size:
                    num_new_samples = len(buffer_data)

                    
                    
                    
                    

                    
                    
                    total_new_samples = num_new_samples * expected_samples
                    dset_data.resize(dset_data.shape[0] + total_new_samples, axis=0)  
                    dset_timestamps.resize(dset_timestamps.shape[0] + num_new_samples, axis=0)  

                    
                    
                    

                    
                    flat_data = np.concatenate(buffer_data)

                    
                    if flat_data.shape[0] != dset_data[-total_new_samples:].shape[0]:
                        
                        continue
                    
                    
                    dset_data[-total_new_samples:] = flat_data  
                    dset_timestamps[-num_new_samples:] = np.array(buffer_timestamps)  

                    
                    buffer_data.clear()
                    buffer_timestamps.clear()

            except queue.Empty:
                continue  
            except Exception as e:
                print(f"Error while saving data: {e}")

    print("Data saving stopped.")


        
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff=300, fs=20000, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    return lfilter(b, a, data)
     
import os
import time
import numpy as np
import queue
import pickle

MAX_MEMORY_BYTES = 10 * 1024 * 1024  
SAVE_DIR = rf"D:\Representation\streamed_data"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_queue_size_bytes(q):
    return sum(item[0].nbytes for item in list(q.queue))

def save_queue_to_disk(q, file_index):
    data_to_save = [(data, ts) for data, ts in list(q.queue)]
    save_path = os.path.join(SAVE_DIR, f"data_chunk_{file_index}.npz")
    np.savez_compressed(save_path, data=data_to_save)
    print(f"Saved data_queue to {save_path}")
    q.queue.clear()

def initialize_streaming(data_queue, timestamp_queue, max_queue_size=10):
    global bioCam, is_streaming, first_chunk_saved, file_counter,start_acq

    
    is_streaming = False

    try:
        
        def on_data_received(sender, e):
            global first_chunk_saved, file_counter,start_acq
            try:
                
                callback_start = time.time()


                
                data = np.frombuffer(e.Payload, dtype=np.uint16).copy()
                
                
                
                
                length_of_data = len(data)  
                
                

         

                if len(data) == 0:
                    print("Warning: Empty data payload received!")
                    return

                
                timestamp = e.Header.Timestamp
                
                
                
                

                
                if not data_queue.empty():
                    _, latest_timestamp = data_queue.queue[-1]  
                    if timestamp == latest_timestamp:
                        print(f"Duplicate timestamp detected: {timestamp}, Payload size={len(data)}")
                        return  


                
                if data_queue.qsize() >= max_queue_size:
                    data_queue.get()  
                data_queue.put((data, timestamp))  

                
                if timestamp_queue.qsize() >= max_queue_size:
                    timestamp_queue.get()  
                timestamp_queue.put(timestamp)  

                
                callback_end = time.time()
                

            except Exception as ex:
                print(f"Error in on_data_received: {ex}")

        
        data_received_event = bioCam.GetType().GetEvent("DataReceived")
        from System import EventHandler
        data_received_delegate = EventHandler[dataPacketReceivedEventArgsType](on_data_received)
        data_received_event.AddEventHandler(bioCam, data_received_delegate)

        
        start_streaming_method = bioCam.GetType().GetMethod("StartDataStreaming")
        start_streaming_method.Invoke(bioCam, [Int32(100), False])  
        start_acq=time.time()
        is_streaming = True
        print("Data streaming started successfully.")

    except Exception as e:
        print(f"Error during BioCAM initialization: {e}")


def create_chcoords(electrodes):
    ch_coords = []
    constructor_2 = chCoordType.GetConstructor([Int32, Int32])  
    for el in electrodes:
        param1 = Int32(el[0])
        param2 = Int32(el[1])
        ch_coord_instance = constructor_2.Invoke([param1, param2])
        ch_coords.append(ch_coord_instance)
    return ch_coords


def create_endpoints(chcoord_list, label_prefix):
    endpoints = []
    endpoint_constructor = StimEndPointDupleXType.GetConstructor([
        chCoordType, String, String, String
    ])

    for idx, ch in enumerate(chcoord_list):
        name = String(f"{label_prefix}_EP_{idx}")
        location = String(f"Index_{idx}")  
        description = String("Auto-generated endpoint")

        endpoint = endpoint_constructor.Invoke([ch, name, location, description])
        endpoints.append(endpoint)

    return endpoints



def save_streamed_data_preallocated(h5_filename, data_queue, prealloc_done, initial_chunks=2460*10, batch_size=10):
   
    from queue import Empty

    global stop_event

    samples_per_chunk = 4096 * 1975  
    total_chunks_allocated = initial_chunks  
    total_chunks_written = 0  

    
    with h5py.File(h5_filename, "w") as h5file:
        
        dset_data = h5file.create_dataset(
            "electrode_data",
            shape=(total_chunks_allocated * samples_per_chunk,),
            maxshape=(None,),  
            dtype='uint16'
        )

        
        dset_timestamps = h5file.create_dataset(
            "timestamps",
            shape=(total_chunks_allocated,),
            maxshape=(None,),
            dtype='int64'
        )

        print(f"Preallocated space for {total_chunks_allocated} chunks.")
        prealloc_done.set()  


        
        buffer_data = []
        buffer_timestamps = []

        
        while not stop_event.is_set():
            try:
                
                data, timestamp = data_queue.get(timeout=1)

                
                if data.shape[0] != samples_per_chunk:
                    print(f"Unexpected data shape: {data.shape[0]}, expected {samples_per_chunk}")
                    continue

                
                buffer_data.append(data)
                buffer_timestamps.append(timestamp)

                
                if len(buffer_data) >= batch_size:
                    num_new_chunks = len(buffer_data)
                    total_new_samples = num_new_chunks * samples_per_chunk

                    
                    if total_chunks_written + num_new_chunks > total_chunks_allocated:
                        new_alloc = total_chunks_allocated * 2
                        dset_data.resize((new_alloc * samples_per_chunk,))
                        dset_timestamps.resize((new_alloc,))
                        total_chunks_allocated = new_alloc
                        print(f"Resized datasets to hold {total_chunks_allocated} chunks.")

                    
                    start_idx = total_chunks_written * samples_per_chunk
                    end_idx = start_idx + total_new_samples

                    dset_data[start_idx:end_idx] = np.concatenate(buffer_data)
                    dset_timestamps[total_chunks_written:total_chunks_written + num_new_chunks] = buffer_timestamps

                    
                    total_chunks_written += num_new_chunks
                    buffer_data.clear()
                    buffer_timestamps.clear()

            except Empty:
                continue  
            except Exception as e:
                print(f"Error while saving data: {e}")

        print("Data saving stopped.")


def pre_train():
    global epoch_id, chip_id,session_type

    global bioCam, stop_event, protocol_manager, meaPlatePilot,start_acq
    
    
    stop_event = Event()
    prealloc_done = Event()
    
    initialize_biocam()
    
    meaPlatePilot = bioCam.MeaPlate

    biocam_settings = bioCam.Settings
    
    
    biocam_settings = bioCam.Settings
    def get_setting(biocam_settings,prop_name):
        prop = biocam_settings.GetType().GetProperty(prop_name)
        return prop.GetValue(biocam_settings)
    def set_setting(biocam_settings,prop_name,value=True):
        prop = biocam_settings.GetType().GetProperty(prop_name)
        return prop.SetValue(biocam_settings,value)





   
    BiocamSettings = bioCam.Settings

    
    hp_pre_enabled = get_setting(BiocamSettings,"IsHpPreEnabled")
    hp_post_enabled = get_setting(BiocamSettings,"IsHpPostEnabled")
    lp_enabled = get_setting(BiocamSettings,"IsLpEnabled")
        
    hp_pre_cutoff = get_setting(BiocamSettings,"HpPreCutoffFrequency")
    hp_post_cutoff = get_setting(BiocamSettings,"HpPostCutOffFrequency")
    hp_post_order = get_setting(BiocamSettings,"HpPostOrder")
    lp_cutoff = get_setting(BiocamSettings,"LpCutoffFrequency")
        
    
    print(f"High-pass pre filter enabled: {hp_pre_enabled}")
    print(f"High-pass pre filter cutoff: {hp_pre_cutoff} Hz")
        
    print(f"High-pass post filter enabled: {hp_post_enabled}")
    print(f"High-pass post filter cutoff: {hp_post_cutoff} Hz")
    print(f"High-pass post filter order: {hp_post_order}")
        
    print(f"Low-pass filter enabled: {lp_enabled}")
    print(f"Low-pass filter cutoff: {lp_cutoff} Hz")
    set_setting(BiocamSettings,"IsHpPreEnabled")
    set_setting(BiocamSettings,"IsHpPostEnabled")

    hp_pre_enabled = get_setting(BiocamSettings,"IsHpPreEnabled")
    hp_post_enabled = get_setting(BiocamSettings,"IsHpPostEnabled")
    lp_enabled = get_setting(BiocamSettings,"IsLpEnabled")
        
    hp_pre_cutoff = get_setting(BiocamSettings,"HpPreCutoffFrequency")
    hp_post_cutoff = get_setting(BiocamSettings,"HpPostCutOffFrequency")
    hp_post_order = get_setting(BiocamSettings,"HpPostOrder")
    lp_cutoff = get_setting(BiocamSettings,"LpCutoffFrequency")
        
    
    print(f"High-pass pre filter enabled: {hp_pre_enabled}")
    print(f"High-pass pre filter cutoff: {hp_pre_cutoff} Hz")
        
    print(f"High-pass post filter enabled: {hp_post_enabled}")
    print(f"High-pass post filter cutoff: {hp_post_cutoff} Hz")
    print(f"High-pass post filter order: {hp_post_order}")
        
    print(f"Low-pass filter enabled: {lp_enabled}")
    print(f"Low-pass filter cutoff: {lp_cutoff} Hz")

    
    
    
    
    
    



    
    

    
    io_signal_settings = biocam_settings.IOSignalsSettings
    
    
    target_signals = [0, 1]  

    for key in io_signal_settings.Keys:
        if int(key) in target_signals:
            
            setting_obj = io_signal_settings[key]
            
            
            setting_obj.GetType().GetProperty("IsOn").SetValue(setting_obj, True)
            
            
            method = setting_obj.GetType().GetMethod("SetOutputChToFirstAvailable")
            method.Invoke(setting_obj, [])
            
            print(f"Enabled and assigned IO signal: {key.ToString()}")
            
            
            output_ch_prop = setting_obj.GetType().GetProperty("OutputCh")
            ch_coord = output_ch_prop.GetValue(setting_obj)
            
            
            row_prop = ch_coord.GetType().GetProperty("Row").GetValue(ch_coord)
            col_prop = ch_coord.GetType().GetProperty("Col").GetValue(ch_coord)
            
            
            
            print(f"{key.ToString()} assigned to OutputCh: Row {row_prop}, Column {col_prop}")
            
    
    set_chamber_temperature(37.0)

    
    data_queue = Queue(maxsize=10)
    timestamp_queue = Queue(maxsize=10)
    initialize_streaming(data_queue, timestamp_queue)

    
    positive_electrodes = [(17, 17), (17, 21), (17, 41), (17, 45), (21, 17), (21, 21), (21, 41), (21, 45), (25, 17), (25, 21), (25, 41), (25, 45), (33, 9), (33, 13), (33, 49), (33, 53), (37, 13), (37, 17), (37, 45), (37, 49), (41, 17), (41, 21), (41, 41), (41, 45), (45, 21), (45, 25), (45, 29), (45, 33), (45, 37), (45, 41)]
    negative_electrodes = [(17, 18), (17, 22), (17, 42), (17, 46), (21, 18), (21, 22), (21, 42), (21, 46), (25, 18), (25, 22), (25, 42), (25, 46), (33, 10), (33, 14), (33, 50), (33, 54), (37, 14), (37, 18), (37, 46), (37, 50), (41, 18), (41, 22), (41, 42), (41, 46), (45, 22), (45, 26), (45, 30), (45, 34), (45, 38), (45, 42)]

    alt_positive_electrodes = [(13, 17), (13, 21), (13, 41), (13, 45), (17, 17), (17, 21), (17, 41), (17, 45), (21, 17), (21, 21), (21, 41), (21, 45), (33, 21), (33, 25), (33, 29), (33, 33), (33, 37), (33, 41), (37, 17), (37, 21), (37, 41), (37, 45), (41, 13), (41, 17), (41, 45), (41, 49), (45, 9), (45, 13), (45, 49), (45, 53)]
    alt_negative_electrodes = [(13, 18), (13, 22), (13, 42), (13, 46), (17, 18), (17, 22), (17, 42), (17, 46), (21, 18), (21, 22), (21, 42), (21, 46), (33, 22), (33, 26), (33, 30), (33, 34), (33, 38), (33, 42), (37, 18), (37, 22), (37, 42), (37, 46), (41, 14), (41, 18), (41, 46), (41, 50), (45, 10), (45, 14), (45, 50), (45, 54)]

    
    positive_coords = create_chcoords(positive_electrodes)
    negative_coords = create_chcoords(negative_electrodes)
    alt_positive_coords = create_chcoords(alt_positive_electrodes)
    alt_negative_coords = create_chcoords(alt_negative_electrodes)

    endpoints_protocol1 = (
        create_endpoints(positive_coords, "Pos"),
        create_endpoints(negative_coords, "Neg")
    )
    
    endpoints_protocol2 = (
        create_endpoints(alt_positive_coords, "Pos_Alt"),
        create_endpoints(alt_negative_coords, "Neg_Alt")
    )

    
    stimulator = bioCam.Stimulator
    stimulator.Initialize()
    stimulator.Start()

    
    stimulatorSettings = stimulator.Settings 
    set_stimulation_calibration(stimulatorSettings)
        
    is_stim_calibration_on = stimulatorSettings.GetType().GetProperty("IsStimCalibrationOn").GetValue(stimulatorSettings)
    print(f"is stim calibration on: {is_stim_calibration_on}")
        
    calibration_distance_us = stimulatorSettings.GetType().GetProperty("CalibrationDistanceMicroSec").GetValue(stimulatorSettings)
    print(f"Calibration distance usec: {calibration_distance_us}")  

    

    protocol_manager = stimulator.Protocol

    
    protocols = []
    for i in range(2):
        stim_props = create_stim_properties()
        pulse = create_rectangular_pulse(
            name=f"BasicPulse_{i}",
            stim_properties=stim_props,
            amp1=positive_amplitude,
            width1=0.5*(1/high_freq_coupling)*1_000_000,
            inter_width=between_phase_offset,
            amp2=negative_amplitude,
            width2=0.5*(1/high_freq_coupling)*1_000_000
        )
        
        protocol = create_basic_protocol(
            name=f"Alternating_{i}",
            pulse=pulse,
            stim_properties=stim_props,
            frequency=1,
            duration=0.05
        )
        
        if i == 0:
            protocol.PositiveEndPoints = endpoints_protocol1[0]
            protocol.NegativeEndPoints = endpoints_protocol1[1]


        else:
            protocol.PositiveEndPoints = endpoints_protocol2[0]
            protocol.NegativeEndPoints = endpoints_protocol2[1]
            
        protocols.append(protocol)



    
    
    
    h5_filename = rf"D:\Representation\streamed_data\{chip_id}_preTrain_e{epoch_id}.h5"
    
    saving_thread = Thread(target=save_streamed_data_preallocated, args=(h5_filename, data_queue,prealloc_done ), daemon=True)
    saving_thread.start()

    print("Waiting for saving thread to be ready...")
    prealloc_done.wait()  
    print("OK, ready! Let's start stimulation.")
    stim_times=[]
    stim_types = []


    try:
        print("Starting alternating stimulation with parallel recording...")
        while(time.time()-start_acq)<=10:
            time.sleep(0.1)
        while not stop_event.is_set():
   
            start_time = time.time()

            stims=[1,1,1,1,1,2,2,2,2,2]
            random.shuffle(stims)
            for stim in stims:

                stim_times.append(time.time()-start_acq)
                stim_types.append(stim)
                
                for count_for_low_freq in range(low_freq_coupling):
                    for count_for_high_freq in range(high_freq_coupling):
                        stimulator.Send(pulse,endpoints_protocol1[0] if stim==1 else endpoints_protocol2[0],endpoints_protocol1[1] if stim==1 else endpoints_protocol2[1])
                        time.sleep(1.0/high_freq_coupling+offset_between_high_and_low_of_biphasic)
                    time.sleep(0.5*(1/low_freq_coupling))
                time.sleep(between_stims_rest)
                
            stop_event.set()
            time.sleep(10*60)

    except KeyboardInterrupt:
        print("Experiment stopped by user")
    finally:
        stop_event.set()
        saving_thread.join()
        
        terminate_acquisition()
        with open(rf'{chip_id}_stim_times_preTrain_e{epoch_id}.pkl', 'wb') as file:
            pickle.dump(stim_times, file)

        with open(rf'{chip_id}_stim_types_preTrain_e{epoch_id}.pkl' , 'wb') as file :
            pickle.dump(stim_types,file)


        print("End of pre-train session")
        

def simple_test():
    global epoch_id, chip_id,session_type

    global bioCam, stop_event, protocol_manager, meaPlatePilot,start_acq
    
    
    stop_event = Event()
    prealloc_done = Event()
    
    initialize_biocam()
    
    meaPlatePilot = bioCam.MeaPlate

    biocam_settings = bioCam.Settings
    
    
    biocam_settings = bioCam.Settings
    def get_setting(biocam_settings,prop_name):
        prop = biocam_settings.GetType().GetProperty(prop_name)
        return prop.GetValue(biocam_settings)
    def set_setting(biocam_settings,prop_name,value=True):
        prop = biocam_settings.GetType().GetProperty(prop_name)
        return prop.SetValue(biocam_settings,value)





   
    BiocamSettings = bioCam.Settings

    
    hp_pre_enabled = get_setting(BiocamSettings,"IsHpPreEnabled")
    hp_post_enabled = get_setting(BiocamSettings,"IsHpPostEnabled")
    lp_enabled = get_setting(BiocamSettings,"IsLpEnabled")
        
    hp_pre_cutoff = get_setting(BiocamSettings,"HpPreCutoffFrequency")
    hp_post_cutoff = get_setting(BiocamSettings,"HpPostCutOffFrequency")
    hp_post_order = get_setting(BiocamSettings,"HpPostOrder")
    lp_cutoff = get_setting(BiocamSettings,"LpCutoffFrequency")
        
    
    print(f"High-pass pre filter enabled: {hp_pre_enabled}")
    print(f"High-pass pre filter cutoff: {hp_pre_cutoff} Hz")
        
    print(f"High-pass post filter enabled: {hp_post_enabled}")
    print(f"High-pass post filter cutoff: {hp_post_cutoff} Hz")
    print(f"High-pass post filter order: {hp_post_order}")
        
    print(f"Low-pass filter enabled: {lp_enabled}")
    print(f"Low-pass filter cutoff: {lp_cutoff} Hz")
    set_setting(BiocamSettings,"IsHpPreEnabled")
    set_setting(BiocamSettings,"IsHpPostEnabled")

    hp_pre_enabled = get_setting(BiocamSettings,"IsHpPreEnabled")
    hp_post_enabled = get_setting(BiocamSettings,"IsHpPostEnabled")
    lp_enabled = get_setting(BiocamSettings,"IsLpEnabled")
        
    hp_pre_cutoff = get_setting(BiocamSettings,"HpPreCutoffFrequency")
    hp_post_cutoff = get_setting(BiocamSettings,"HpPostCutOffFrequency")
    hp_post_order = get_setting(BiocamSettings,"HpPostOrder")
    lp_cutoff = get_setting(BiocamSettings,"LpCutoffFrequency")
        
    
    print(f"High-pass pre filter enabled: {hp_pre_enabled}")
    print(f"High-pass pre filter cutoff: {hp_pre_cutoff} Hz")
        
    print(f"High-pass post filter enabled: {hp_post_enabled}")
    print(f"High-pass post filter cutoff: {hp_post_cutoff} Hz")
    print(f"High-pass post filter order: {hp_post_order}")
        
    print(f"Low-pass filter enabled: {lp_enabled}")
    print(f"Low-pass filter cutoff: {lp_cutoff} Hz")

    
    
    
    
    
    



    
    

    
    io_signal_settings = biocam_settings.IOSignalsSettings
    
    
    target_signals = [0, 1]  

    for key in io_signal_settings.Keys:
        if int(key) in target_signals:
            
            setting_obj = io_signal_settings[key]
            
            
            setting_obj.GetType().GetProperty("IsOn").SetValue(setting_obj, True)
            
            
            method = setting_obj.GetType().GetMethod("SetOutputChToFirstAvailable")
            method.Invoke(setting_obj, [])
            
            print(f"Enabled and assigned IO signal: {key.ToString()}")
            
            
            output_ch_prop = setting_obj.GetType().GetProperty("OutputCh")
            ch_coord = output_ch_prop.GetValue(setting_obj)
            
            
            row_prop = ch_coord.GetType().GetProperty("Row").GetValue(ch_coord)
            col_prop = ch_coord.GetType().GetProperty("Col").GetValue(ch_coord)
            
            
            
            print(f"{key.ToString()} assigned to OutputCh: Row {row_prop}, Column {col_prop}")
            
    
    set_chamber_temperature(37.0)

    
    data_queue = Queue(maxsize=10)
    timestamp_queue = Queue(maxsize=10)
    initialize_streaming(data_queue, timestamp_queue)

    
    positive_electrodes = [(17, 17), (17, 21), (17, 41), (17, 45), (21, 17), (21, 21), (21, 41), (21, 45), (25, 17), (25, 21), (25, 41), (25, 45), (33, 9), (33, 13), (33, 49), (33, 53), (37, 13), (37, 17), (37, 45), (37, 49), (41, 17), (41, 21), (41, 41), (41, 45), (45, 21), (45, 25), (45, 29), (45, 33), (45, 37), (45, 41)]
    negative_electrodes = [(17, 18), (17, 22), (17, 42), (17, 46), (21, 18), (21, 22), (21, 42), (21, 46), (25, 18), (25, 22), (25, 42), (25, 46), (33, 10), (33, 14), (33, 50), (33, 54), (37, 14), (37, 18), (37, 46), (37, 50), (41, 18), (41, 22), (41, 42), (41, 46), (45, 22), (45, 26), (45, 30), (45, 34), (45, 38), (45, 42)]

    alt_positive_electrodes = [(13, 17), (13, 21), (13, 41), (13, 45), (17, 17), (17, 21), (17, 41), (17, 45), (21, 17), (21, 21), (21, 41), (21, 45), (33, 21), (33, 25), (33, 29), (33, 33), (33, 37), (33, 41), (37, 17), (37, 21), (37, 41), (37, 45), (41, 13), (41, 17), (41, 45), (41, 49), (45, 9), (45, 13), (45, 49), (45, 53)]
    alt_negative_electrodes = [(13, 18), (13, 22), (13, 42), (13, 46), (17, 18), (17, 22), (17, 42), (17, 46), (21, 18), (21, 22), (21, 42), (21, 46), (33, 22), (33, 26), (33, 30), (33, 34), (33, 38), (33, 42), (37, 18), (37, 22), (37, 42), (37, 46), (41, 14), (41, 18), (41, 46), (41, 50), (45, 10), (45, 14), (45, 50), (45, 54)]

    
    positive_coords = create_chcoords(positive_electrodes)
    negative_coords = create_chcoords(negative_electrodes)
    alt_positive_coords = create_chcoords(alt_positive_electrodes)
    alt_negative_coords = create_chcoords(alt_negative_electrodes)

    endpoints_protocol1 = (
        create_endpoints(positive_coords, "Pos"),
        create_endpoints(negative_coords, "Neg")
    )
    
    endpoints_protocol2 = (
        create_endpoints(alt_positive_coords, "Pos_Alt"),
        create_endpoints(alt_negative_coords, "Neg_Alt")
    )

    
    stimulator = bioCam.Stimulator
    stimulator.Initialize()
    stimulator.Start()

    
    stimulatorSettings = stimulator.Settings 
    set_stimulation_calibration(stimulatorSettings)
        
    is_stim_calibration_on = stimulatorSettings.GetType().GetProperty("IsStimCalibrationOn").GetValue(stimulatorSettings)
    print(f"is stim calibration on: {is_stim_calibration_on}")
        
    calibration_distance_us = stimulatorSettings.GetType().GetProperty("CalibrationDistanceMicroSec").GetValue(stimulatorSettings)
    print(f"Calibration distance usec: {calibration_distance_us}")  

    

    protocol_manager = stimulator.Protocol

    
    protocols = []
    for i in range(2):
        stim_props = create_stim_properties()
        pulse = create_rectangular_pulse(
            name=f"BasicPulse_{i}",
            stim_properties=stim_props,
            amp1=positive_amplitude,
            width1=0.5*(1/high_freq_coupling)*1_000_000,
            inter_width=between_phase_offset,
            amp2=negative_amplitude,
            width2=0.5*(1/high_freq_coupling)*1_000_000
        )
        
        protocol = create_basic_protocol(
            name=f"Alternating_{i}",
            pulse=pulse,
            stim_properties=stim_props,
            frequency=1,
            duration=0.05
        )
        
        if i == 0:
            protocol.PositiveEndPoints = endpoints_protocol1[0]
            protocol.NegativeEndPoints = endpoints_protocol1[1]


        else:
            protocol.PositiveEndPoints = endpoints_protocol2[0]
            protocol.NegativeEndPoints = endpoints_protocol2[1]
            
        protocols.append(protocol)



    
    
    
    h5_filename = rf"D:\Representation\streamed_data\{chip_id}_preTrain_e{epoch_id}.h5"
    
    saving_thread = Thread(target=save_streamed_data_preallocated, args=(h5_filename, data_queue,prealloc_done ), daemon=True)
    saving_thread.start()

    print("Waiting for saving thread to be ready...")
    prealloc_done.wait()  
    print("OK, ready! Let's start stimulation.")
    stim_times=[]
    stim_types = []


    try:
        print("Starting alternating stimulation with parallel recording...")
        while(time.time()-start_acq)<=10:
            time.sleep(0.1)
        while not stop_event.is_set():
   
            start_time = time.time()

            stims=[1,1,1,1,1,2,2,2,2,2]
            random.shuffle(stims)
            for stim in stims:

                stim_times.append(time.time()-start_acq)
                stim_types.append(stim)
                
                for count_for_low_freq in range(5):
                    for count_for_high_freq in range(5):
                        stimulator.Send(pulse,endpoints_protocol1[0] if stim==1 else endpoints_protocol2[0],endpoints_protocol1[1] if stim==1 else endpoints_protocol2[1])
                        time.sleep(1.0/high_freq_coupling+offset_between_high_and_low_of_biphasic)
                    time.sleep(0.2)
                time.sleep(10)
                
            stop_event.set()
            time.sleep(10)

    except KeyboardInterrupt:
        print("Experiment stopped by user")
    finally:
        stop_event.set()
        saving_thread.join()
        
        terminate_acquisition()
        with open(rf'{chip_id}_stim_times_preTrain_e{epoch_id}.pkl', 'wb') as file:
            pickle.dump(stim_times, file)

        with open(rf'{chip_id}_stim_types_preTrain_e{epoch_id}.pkl' , 'wb') as file :
            pickle.dump(stim_types,file)


        print("End of pre-train session")
        
        
def post_train():
    global epoch_id, chip_id,session_type

    global bioCam, stop_event, protocol_manager, meaPlatePilot,start_acq
    
    
    stop_event = Event()
    prealloc_done = Event()
    
    initialize_biocam()
    
    meaPlatePilot = bioCam.MeaPlate

    biocam_settings = bioCam.Settings
    
    
    biocam_settings = bioCam.Settings
    def get_setting(biocam_settings,prop_name):
        prop = biocam_settings.GetType().GetProperty(prop_name)
        return prop.GetValue(biocam_settings)
    def set_setting(biocam_settings,prop_name,value=True):
        prop = biocam_settings.GetType().GetProperty(prop_name)
        return prop.SetValue(biocam_settings,value)





   
    BiocamSettings = bioCam.Settings

    
    hp_pre_enabled = get_setting(BiocamSettings,"IsHpPreEnabled")
    hp_post_enabled = get_setting(BiocamSettings,"IsHpPostEnabled")
    lp_enabled = get_setting(BiocamSettings,"IsLpEnabled")
        
    hp_pre_cutoff = get_setting(BiocamSettings,"HpPreCutoffFrequency")
    hp_post_cutoff = get_setting(BiocamSettings,"HpPostCutOffFrequency")
    hp_post_order = get_setting(BiocamSettings,"HpPostOrder")
    lp_cutoff = get_setting(BiocamSettings,"LpCutoffFrequency")
        
    
    print(f"High-pass pre filter enabled: {hp_pre_enabled}")
    print(f"High-pass pre filter cutoff: {hp_pre_cutoff} Hz")
        
    print(f"High-pass post filter enabled: {hp_post_enabled}")
    print(f"High-pass post filter cutoff: {hp_post_cutoff} Hz")
    print(f"High-pass post filter order: {hp_post_order}")
        
    print(f"Low-pass filter enabled: {lp_enabled}")
    print(f"Low-pass filter cutoff: {lp_cutoff} Hz")
    set_setting(BiocamSettings,"IsHpPreEnabled")
    set_setting(BiocamSettings,"IsHpPostEnabled")

    hp_pre_enabled = get_setting(BiocamSettings,"IsHpPreEnabled")
    hp_post_enabled = get_setting(BiocamSettings,"IsHpPostEnabled")
    lp_enabled = get_setting(BiocamSettings,"IsLpEnabled")
        
    hp_pre_cutoff = get_setting(BiocamSettings,"HpPreCutoffFrequency")
    hp_post_cutoff = get_setting(BiocamSettings,"HpPostCutOffFrequency")
    hp_post_order = get_setting(BiocamSettings,"HpPostOrder")
    lp_cutoff = get_setting(BiocamSettings,"LpCutoffFrequency")
        
    
    print(f"High-pass pre filter enabled: {hp_pre_enabled}")
    print(f"High-pass pre filter cutoff: {hp_pre_cutoff} Hz")
        
    print(f"High-pass post filter enabled: {hp_post_enabled}")
    print(f"High-pass post filter cutoff: {hp_post_cutoff} Hz")
    print(f"High-pass post filter order: {hp_post_order}")
        
    print(f"Low-pass filter enabled: {lp_enabled}")
    print(f"Low-pass filter cutoff: {lp_cutoff} Hz")

    
    
    
    
    
    



    
    

    
    io_signal_settings = biocam_settings.IOSignalsSettings
    
    
    target_signals = [0, 1]  

    for key in io_signal_settings.Keys:
        if int(key) in target_signals:
            
            setting_obj = io_signal_settings[key]
            
            
            setting_obj.GetType().GetProperty("IsOn").SetValue(setting_obj, True)
            
            
            method = setting_obj.GetType().GetMethod("SetOutputChToFirstAvailable")
            method.Invoke(setting_obj, [])
            
            print(f"Enabled and assigned IO signal: {key.ToString()}")
            
            
            output_ch_prop = setting_obj.GetType().GetProperty("OutputCh")
            ch_coord = output_ch_prop.GetValue(setting_obj)
            
            
            row_prop = ch_coord.GetType().GetProperty("Row").GetValue(ch_coord)
            col_prop = ch_coord.GetType().GetProperty("Col").GetValue(ch_coord)
            
            
            
            print(f"{key.ToString()} assigned to OutputCh: Row {row_prop}, Column {col_prop}")
            
    
    set_chamber_temperature(37.0)

    
    data_queue = Queue(maxsize=10)
    timestamp_queue = Queue(maxsize=10)
    initialize_streaming(data_queue, timestamp_queue)

    
    positive_electrodes = [(17, 17), (17, 21), (17, 41), (17, 45), (21, 17), (21, 21), (21, 41), (21, 45), (25, 17), (25, 21), (25, 41), (25, 45), (33, 9), (33, 13), (33, 49), (33, 53), (37, 13), (37, 17), (37, 45), (37, 49), (41, 17), (41, 21), (41, 41), (41, 45), (45, 21), (45, 25), (45, 29), (45, 33), (45, 37), (45, 41)]
    negative_electrodes = [(17, 18), (17, 22), (17, 42), (17, 46), (21, 18), (21, 22), (21, 42), (21, 46), (25, 18), (25, 22), (25, 42), (25, 46), (33, 10), (33, 14), (33, 50), (33, 54), (37, 14), (37, 18), (37, 46), (37, 50), (41, 18), (41, 22), (41, 42), (41, 46), (45, 22), (45, 26), (45, 30), (45, 34), (45, 38), (45, 42)]

    alt_positive_electrodes = [(13, 17), (13, 21), (13, 41), (13, 45), (17, 17), (17, 21), (17, 41), (17, 45), (21, 17), (21, 21), (21, 41), (21, 45), (33, 21), (33, 25), (33, 29), (33, 33), (33, 37), (33, 41), (37, 17), (37, 21), (37, 41), (37, 45), (41, 13), (41, 17), (41, 45), (41, 49), (45, 9), (45, 13), (45, 49), (45, 53)]
    alt_negative_electrodes = [(13, 18), (13, 22), (13, 42), (13, 46), (17, 18), (17, 22), (17, 42), (17, 46), (21, 18), (21, 22), (21, 42), (21, 46), (33, 22), (33, 26), (33, 30), (33, 34), (33, 38), (33, 42), (37, 18), (37, 22), (37, 42), (37, 46), (41, 14), (41, 18), (41, 46), (41, 50), (45, 10), (45, 14), (45, 50), (45, 54)]

    
    positive_coords = create_chcoords(positive_electrodes)
    negative_coords = create_chcoords(negative_electrodes)
    alt_positive_coords = create_chcoords(alt_positive_electrodes)
    alt_negative_coords = create_chcoords(alt_negative_electrodes)

    endpoints_protocol1 = (
        create_endpoints(positive_coords, "Pos"),
        create_endpoints(negative_coords, "Neg")
    )
    
    endpoints_protocol2 = (
        create_endpoints(alt_positive_coords, "Pos_Alt"),
        create_endpoints(alt_negative_coords, "Neg_Alt")
    )

    
    stimulator = bioCam.Stimulator
    stimulator.Initialize()
    stimulator.Start()

    
    stimulatorSettings = stimulator.Settings 
    set_stimulation_calibration(stimulatorSettings)
        
    is_stim_calibration_on = stimulatorSettings.GetType().GetProperty("IsStimCalibrationOn").GetValue(stimulatorSettings)
    print(f"is stim calibration on: {is_stim_calibration_on}")
        
    calibration_distance_us = stimulatorSettings.GetType().GetProperty("CalibrationDistanceMicroSec").GetValue(stimulatorSettings)
    print(f"Calibration distance usec: {calibration_distance_us}")  

    

    protocol_manager = stimulator.Protocol

    
    protocols = []
    for i in range(2):
        stim_props = create_stim_properties()
        pulse = create_rectangular_pulse(
            name=f"BasicPulse_{i}",
            stim_properties=stim_props,
            amp1=positive_amplitude,
            width1=0.5*(1/high_freq_coupling)*1_000_000,
            inter_width=between_phase_offset,
            amp2=negative_amplitude,
            width2=0.5*(1/high_freq_coupling)*1_000_000
        )
        
        protocol = create_basic_protocol(
            name=f"Alternating_{i}",
            pulse=pulse,
            stim_properties=stim_props,
            frequency=1,
            duration=0.05
        )
        
        if i == 0:
            protocol.PositiveEndPoints = endpoints_protocol1[0]
            protocol.NegativeEndPoints = endpoints_protocol1[1]


        else:
            protocol.PositiveEndPoints = endpoints_protocol2[0]
            protocol.NegativeEndPoints = endpoints_protocol2[1]
            
        protocols.append(protocol)



    
    
    
    h5_filename = rf"D:\Representation\streamed_data\{chip_id}_postTrain_e{epoch_id}.h5"
    
    saving_thread = Thread(target=save_streamed_data_preallocated, args=(h5_filename, data_queue,prealloc_done ), daemon=True)
    saving_thread.start()

    print("Waiting for saving thread to be ready...")
    prealloc_done.wait()  
    print("OK, ready! Let's start stimulation.")
    stim_times=[]
    stim_types = []


    try:
        print("Starting alternating stimulation with parallel recording...")
        while(time.time()-start_acq)<=10:
            time.sleep(0.1)
        while not stop_event.is_set():
   
            start_time = time.time()
            time.sleep(30*60)

            stims=[1,1,1,1,1,2,2,2,2,2]
            random.shuffle(stims)
            for stim in stims:

                stim_times.append(time.time()-start_acq)
                stim_types.append(stim)
                
                for count_for_low_freq in range(low_freq_coupling):
                    for count_for_high_freq in range(high_freq_coupling):
                        stimulator.Send(pulse,endpoints_protocol1[0] if stim==1 else endpoints_protocol2[0],endpoints_protocol1[1] if stim==1 else endpoints_protocol2[1])
                        time.sleep(1.0/high_freq_coupling+offset_between_high_and_low_of_biphasic)
                    time.sleep(0.5*(1/low_freq_coupling))
                time.sleep(between_stims_rest)
                
            stop_event.set()

    except KeyboardInterrupt:
        print("Experiment stopped by user")
    finally:
        stop_event.set()
        saving_thread.join()
        
        terminate_acquisition()
        with open(rf'{chip_id}_stim_times_postTrain_e{epoch_id}.pkl', 'wb') as file:
            pickle.dump(stim_times, file)

        with open(rf'{chip_id}_stim_types_postTrain_e{epoch_id}.pkl' , 'wb') as file :
            pickle.dump(stim_types,file)


        print("End of post-train session")

def train():
    global epoch_id, chip_id,session_type

    global bioCam, stop_event, protocol_manager, meaPlatePilot,start_acq
    
    
    stop_event = Event()
    prealloc_done = Event()
    
    initialize_biocam()
    
    meaPlatePilot = bioCam.MeaPlate

    biocam_settings = bioCam.Settings
    
    
    biocam_settings = bioCam.Settings
    def get_setting(biocam_settings,prop_name):
        prop = biocam_settings.GetType().GetProperty(prop_name)
        return prop.GetValue(biocam_settings)
    def set_setting(biocam_settings,prop_name,value=True):
        prop = biocam_settings.GetType().GetProperty(prop_name)
        return prop.SetValue(biocam_settings,value)





   
    BiocamSettings = bioCam.Settings

    
    hp_pre_enabled = get_setting(BiocamSettings,"IsHpPreEnabled")
    hp_post_enabled = get_setting(BiocamSettings,"IsHpPostEnabled")
    lp_enabled = get_setting(BiocamSettings,"IsLpEnabled")
        
    hp_pre_cutoff = get_setting(BiocamSettings,"HpPreCutoffFrequency")
    hp_post_cutoff = get_setting(BiocamSettings,"HpPostCutOffFrequency")
    hp_post_order = get_setting(BiocamSettings,"HpPostOrder")
    lp_cutoff = get_setting(BiocamSettings,"LpCutoffFrequency")
        
    
    print(f"High-pass pre filter enabled: {hp_pre_enabled}")
    print(f"High-pass pre filter cutoff: {hp_pre_cutoff} Hz")
        
    print(f"High-pass post filter enabled: {hp_post_enabled}")
    print(f"High-pass post filter cutoff: {hp_post_cutoff} Hz")
    print(f"High-pass post filter order: {hp_post_order}")
        
    print(f"Low-pass filter enabled: {lp_enabled}")
    print(f"Low-pass filter cutoff: {lp_cutoff} Hz")
    set_setting(BiocamSettings,"IsHpPreEnabled")
    set_setting(BiocamSettings,"IsHpPostEnabled")

    hp_pre_enabled = get_setting(BiocamSettings,"IsHpPreEnabled")
    hp_post_enabled = get_setting(BiocamSettings,"IsHpPostEnabled")
    lp_enabled = get_setting(BiocamSettings,"IsLpEnabled")
        
    hp_pre_cutoff = get_setting(BiocamSettings,"HpPreCutoffFrequency")
    hp_post_cutoff = get_setting(BiocamSettings,"HpPostCutOffFrequency")
    hp_post_order = get_setting(BiocamSettings,"HpPostOrder")
    lp_cutoff = get_setting(BiocamSettings,"LpCutoffFrequency")
        
    
    print(f"High-pass pre filter enabled: {hp_pre_enabled}")
    print(f"High-pass pre filter cutoff: {hp_pre_cutoff} Hz")
        
    print(f"High-pass post filter enabled: {hp_post_enabled}")
    print(f"High-pass post filter cutoff: {hp_post_cutoff} Hz")
    print(f"High-pass post filter order: {hp_post_order}")
        
    print(f"Low-pass filter enabled: {lp_enabled}")
    print(f"Low-pass filter cutoff: {lp_cutoff} Hz")

    
    
    
    
    
    



    
    

    
    io_signal_settings = biocam_settings.IOSignalsSettings
    
    
    target_signals = [0, 1]  

    for key in io_signal_settings.Keys:
        if int(key) in target_signals:
            
            setting_obj = io_signal_settings[key]
            
            
            setting_obj.GetType().GetProperty("IsOn").SetValue(setting_obj, True)
            
            
            method = setting_obj.GetType().GetMethod("SetOutputChToFirstAvailable")
            method.Invoke(setting_obj, [])
            
            print(f"Enabled and assigned IO signal: {key.ToString()}")
            
            
            output_ch_prop = setting_obj.GetType().GetProperty("OutputCh")
            ch_coord = output_ch_prop.GetValue(setting_obj)
            
            
            row_prop = ch_coord.GetType().GetProperty("Row").GetValue(ch_coord)
            col_prop = ch_coord.GetType().GetProperty("Col").GetValue(ch_coord)
            
            
            
            print(f"{key.ToString()} assigned to OutputCh: Row {row_prop}, Column {col_prop}")
            
    
    set_chamber_temperature(37.0)

    
    data_queue = Queue(maxsize=10)
    timestamp_queue = Queue(maxsize=10)
    initialize_streaming(data_queue, timestamp_queue)

    
    positive_electrodes = [(17, 17), (17, 21), (17, 41), (17, 45), (21, 17), (21, 21), (21, 41), (21, 45), (25, 17), (25, 21), (25, 41), (25, 45), (33, 9), (33, 13), (33, 49), (33, 53), (37, 13), (37, 17), (37, 45), (37, 49), (41, 17), (41, 21), (41, 41), (41, 45), (45, 21), (45, 25), (45, 29), (45, 33), (45, 37), (45, 41)]
    negative_electrodes = [(17, 18), (17, 22), (17, 42), (17, 46), (21, 18), (21, 22), (21, 42), (21, 46), (25, 18), (25, 22), (25, 42), (25, 46), (33, 10), (33, 14), (33, 50), (33, 54), (37, 14), (37, 18), (37, 46), (37, 50), (41, 18), (41, 22), (41, 42), (41, 46), (45, 22), (45, 26), (45, 30), (45, 34), (45, 38), (45, 42)]

    alt_positive_electrodes = [(13, 17), (13, 21), (13, 41), (13, 45), (17, 17), (17, 21), (17, 41), (17, 45), (21, 17), (21, 21), (21, 41), (21, 45), (33, 21), (33, 25), (33, 29), (33, 33), (33, 37), (33, 41), (37, 17), (37, 21), (37, 41), (37, 45), (41, 13), (41, 17), (41, 45), (41, 49), (45, 9), (45, 13), (45, 49), (45, 53)]
    alt_negative_electrodes = [(13, 18), (13, 22), (13, 42), (13, 46), (17, 18), (17, 22), (17, 42), (17, 46), (21, 18), (21, 22), (21, 42), (21, 46), (33, 22), (33, 26), (33, 30), (33, 34), (33, 38), (33, 42), (37, 18), (37, 22), (37, 42), (37, 46), (41, 14), (41, 18), (41, 46), (41, 50), (45, 10), (45, 14), (45, 50), (45, 54)]

    
    positive_coords = create_chcoords(positive_electrodes)
    negative_coords = create_chcoords(negative_electrodes)
    alt_positive_coords = create_chcoords(alt_positive_electrodes)
    alt_negative_coords = create_chcoords(alt_negative_electrodes)

    endpoints_protocol1 = (
        create_endpoints(positive_coords, "Pos"),
        create_endpoints(negative_coords, "Neg")
    )
    
    endpoints_protocol2 = (
        create_endpoints(alt_positive_coords, "Pos_Alt"),
        create_endpoints(alt_negative_coords, "Neg_Alt")
    )

    
    stimulator = bioCam.Stimulator
    stimulator.Initialize()
    stimulator.Start()

    
    stimulatorSettings = stimulator.Settings 
    set_stimulation_calibration(stimulatorSettings)
        
    is_stim_calibration_on = stimulatorSettings.GetType().GetProperty("IsStimCalibrationOn").GetValue(stimulatorSettings)
    print(f"is stim calibration on: {is_stim_calibration_on}")
        
    calibration_distance_us = stimulatorSettings.GetType().GetProperty("CalibrationDistanceMicroSec").GetValue(stimulatorSettings)
    print(f"Calibration distance usec: {calibration_distance_us}")  

    

    protocol_manager = stimulator.Protocol

    
    protocols = []
    for i in range(2):
        stim_props = create_stim_properties()
        pulse = create_rectangular_pulse(
            name=f"BasicPulse_{i}",
            stim_properties=stim_props,
            amp1=positive_amplitude,
            width1=0.5*(1/high_freq_coupling)*1_000_000,
            inter_width=between_phase_offset,
            amp2=negative_amplitude,
            width2=0.5*(1/high_freq_coupling)*1_000_000
        )
        
        protocol = create_basic_protocol(
            name=f"Alternating_{i}",
            pulse=pulse,
            stim_properties=stim_props,
            frequency=1,
            duration=0.05
        )
        
        if i == 0:
            protocol.PositiveEndPoints = endpoints_protocol1[0]
            protocol.NegativeEndPoints = endpoints_protocol1[1]


        else:
            protocol.PositiveEndPoints = endpoints_protocol2[0]
            protocol.NegativeEndPoints = endpoints_protocol2[1]
            
        protocols.append(protocol)



    
    
    
    h5_filename = rf"D:\Representation\streamed_data\{chip_id}_train_e{epoch_id}.h5"
    
    saving_thread = Thread(target=save_streamed_data_preallocated, args=(h5_filename, data_queue,prealloc_done ), daemon=True)
    saving_thread.start()

    print("Waiting for saving thread to be ready...")
    prealloc_done.wait()  
    print("OK, ready! Let's start stimulation.")
    stim_times=[]
    stim_types = []


    try:
        print("Starting alternating stimulation with parallel recording...")
        while(time.time()-start_acq)<=10:
            time.sleep(0.1)
        while not stop_event.is_set():
   


            start_time = time.time()
            for cycle in range(num_cycles):
                for trial in range(num_trials_in_each_cycle):
                    stims=[1,1,1,1,1,2,2,2,2,2]
                    random.shuffle(stims)
                    for stim in stims:

                        stim_times.append(time.time()-start_acq)
                        stim_types.append(stim)
                        
                        for count_for_low_freq in range(low_freq_coupling):
                            for count_for_high_freq in range(high_freq_coupling):
                                stimulator.Send(pulse,endpoints_protocol1[0] if stim==1 else endpoints_protocol2[0],endpoints_protocol1[1] if stim==1 else endpoints_protocol2[1])
                                time.sleep(1.0/high_freq_coupling+offset_between_high_and_low_of_biphasic)
                            time.sleep(0.5*(1/low_freq_coupling))
                        time.sleep(between_stims_rest)
                    time.sleep(between_trials_rest)
                time.sleep(between_cycles_rest)
                    
                
            stop_event.set()

    except KeyboardInterrupt:
        print("Experiment stopped by user")
    finally:
        stop_event.set()
        saving_thread.join()
        
        terminate_acquisition()
        with open(rf'{chip_id}_stim_times_train_e{epoch_id}.pkl', 'wb') as file:
            pickle.dump(stim_times, file)

        with open(rf'{chip_id}_stim_types_train_e{epoch_id}.pkl' , 'wb') as file :
            pickle.dump(stim_types,file)


        print("End of train session")

def create_stim_properties():
    constructor = StimPropertiesType.GetConstructor([
        Int32,Int32,Int64,Int64, Double, Int32, Int32, Int32
    ])
    return constructor.Invoke([
        Int32(1),  
        Int32(425),
        Int64(0),
        Int64(425),
        Double(amplitude_resolution_value),  
        Int32(-25),  
        Int32(25),  
        Int32(1)  
        
    ])

def create_rectangular_pulse(name, stim_properties, amp1, width1, inter_width, amp2, width2):
    constructor = rectangularStimPulseType.GetConstructor([
        String, StimPropertiesType, Double, Int32, Int32, Double, Int32
    ])
    return constructor.Invoke([
        String(name),
        stim_properties,
        Double(amp1),
        Int32(width1),
        Int32(inter_width),
        Double(amp2),
        Int32(width2)
    ])

def create_basic_protocol(name, pulse, stim_properties, frequency, duration):
    pulse_rate = Double(frequency)
    duration_us = Int32(int(duration * 1e6))
    
    constructor = StimTrainProtocolType.GetConstructor([
        String, rectangularStimPulseType, StimPropertiesType, Int32, Double
    ])
    
    protocol = constructor.Invoke([
        String(name),
        pulse,
        stim_properties,
        Int32(1),  
        pulse_rate
    ])

    
    return protocol
if __name__ == "__main__":
    # pre_train()
    # train()
    # post_train()
    simple_test()
