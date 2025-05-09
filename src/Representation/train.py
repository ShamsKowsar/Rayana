

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 22:31:46 2024

@author: elahe
"""


import clr
import random
import os
import time
import numpy as np
from scipy.signal import butter, lfilter
from threading import Thread, Event, Lock
from queue import Queue
import matplotlib.pyplot as plt
import System
from System import Int32, Int64, Double, String
import queue
from matplotlib.colorbar import Colorbar
import clr
import random
import os
import time
import numpy as np
import scipy.io
import h5py  # Import for saving data
from threading import Thread, Event
from queue import Queue
import queue
import System
from System import Int32, Single
import random
from datetime import datetime


program_start = time.time()
# Constants for the Pong Game
FPS = 10  # Frames per second


game_state_lock = Lock()
stop_event = Event()


# Constants for motor regions
#M1 = [(x, y) for x in range(30, 56) for y in range(5, 26)]  # M1 region
#M2 = [(x, y) for x in range(30, 56) for y in range(39, 60)]  # M2 region

# Define indices for M1 and M2 regions
rows_m1 = np.arange(51, 59)  # MATLAB's 30:55 is Python's 30:56
cols_m1 = np.arange(13, 21)
r_m1 = np.repeat(rows_m1, len(cols_m1))
c_m1 = np.tile(cols_m1, len(rows_m1))
M1 = np.ravel_multi_index((r_m1, c_m1), (64, 64))

cols_m2 = np.arange(53, 61)
r_m2 = np.repeat(rows_m1, len(cols_m2))
c_m2 = np.tile(cols_m2, len(rows_m1))
M2 = np.ravel_multi_index((r_m2, c_m2), (64, 64))



# Define sensory electrodes for positive side
positive_electrodes_happy = [
    (5, 5), (5, 23), (5, 41), (5, 59),  # Row 5
    (10, 3), (10, 21), (10, 39), (10, 57)  # Row 10
    ]   
    
# Define negative electrodes (shifted by one column to the right)
negative_electrodes_happy = [
    (5, 6), (5, 24), (5, 42), (5, 60),  # Row 5
    (10, 4), (10, 22), (10, 40), (10, 58)  # Row 10
    ]

positive_electrodes_sad = [
    (5, 5), (5, 23), (5, 41), (5, 59),  # Row 5
    (10, 3), (10, 21), (10, 39), (10, 57)  # Row 10
    ]   
    
# Define negative electrodes (shifted by one column to the right)
negative_electrodes_sad = [
    (5, 6), (5, 24), (5, 42), (5, 60),  # Row 5
    (10, 4), (10, 22), (10, 40), (10, 58)  # Row 10
    ]

# Define constants for pulse widths and inter-pulse width for "create_stimulation_protocols" 
W1 = 200             # Phase width for the 1st phase (us)
W2 = 200             # Phase width for the 2nd phase (us)
IW = 25              # Inter-pulse width (us)

# Define the pulse types and corresponding amplitudes for "create_stimulation_protocols"
pulse_types = ["happy", "sad",]
pulse_amplitude = [
    [15.0, -15.0],  # Reward pulse amplitude pairs (Amp1, Amp2) (uA)
    [15.0, -15.0]   # Punishment pulse amplitude pairs (Amp1, Amp2) (uA)
    ]    

positive_endpoints_full = []
negative_endpoints_full = []


# Load DLLs for MEA API
base_path = r"C:\Users\BioCAM User\Downloads\API\BioCamDriverAPI_v2.5\BioCamDriverAPI_v2.5\API"
clr.AddReference(os.path.join(base_path, "3Brain.BioCamDriver.dll"))
clr.AddReference(os.path.join(base_path, "3Brain.Common.dll"))
# clr.AddReference(os.path.join(base_path, "Newtonsoft.Json.dll"))


# Access the BioCamPool and ChCoord classes through reflection
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


# Global variables for BioCam
bioCam = None
protocol_manager = None
is_streaming = False
meaPlatePilot = None

# Global variable for spike counts
spike_counts = None
slot_index = None
latest_timestamp_us = 0  # Initialize latest_timestamp_us
result = [None,None]
DEBUG = False

# Constants 
# Access the Default field (this is a static field, so no instance needed)
default_stim_properties_field = StimPropertiesType.GetField("Default")

# Get the value of the Default field
default_stim_properties = default_stim_properties_field.GetValue(None)

# Access the 'TimeResolution' property
time_resolution_property = StimPropertiesType.GetProperty("TimeResolutionMicroSec")
time_resolution_value = time_resolution_property.GetValue(default_stim_properties)

# Access the 'AmplitudeResolution' property
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
            
        
        # Retrieve slot indexes for connected BioCAMs
        get_slot_indexes_method = bioCamPoolType.GetMethod("GetSlotIndexesConnectedBioCam")
        connected_indexes = get_slot_indexes_method.Invoke(None, [])
        python_indexes = list(connected_indexes)  # Convert to Python list
        if len(python_indexes) == 0:
            print("Failed to retrieve connected BioCam slot indexes. Exiting...")
            return
        slot_index = python_indexes[0]
        slot_index = Int32(slot_index)  # Convert slot index to System.Int32
        print(f"BioCam device is in slot index: {slot_index}")
        
        print("BioCam successfully initialized.")
    except Exception as e:
        print(f"Error during BioCam initialization: {e}")
        bioCam = None
       

# Function to set chamber temperature to a target value
def set_chamber_temperature(target_temperature_celsius):
    global bioCam, slot_index, meaPlatePilot
    if bioCam is None:
        print("Error: BioCam not initialized. Please initialize BioCam first.")
        return
    
    try:
       
        
        if meaPlatePilot is None:
            print("Error: No IMeaPlatePilot found!")
            return
        
        # Access MeaPlateSettings from IMeaPlatePilot
        settings_property = meaPlatePilot.GetType().GetProperty("Settings")
        plate_settings = settings_property.GetValue(meaPlatePilot)
        
        if plate_settings is None:
            print("Error: No MeaPlateSettings found!")
            return
        
        
        
        # Check if temperature control is on
        is_temp_control_on_property = plate_settings.GetType().GetProperty("IsChamberTemperatureControlOn")
        is_temp_control_on = is_temp_control_on_property.GetValue(plate_settings)

        print(f"Chamber temperature control is {'ON' if is_temp_control_on else 'OFF'}.")

        # If it's OFF, turn it ON
        if not is_temp_control_on:
             is_temp_control_on_property.SetValue(plate_settings, True)
             print("Chamber temperature control has been turned ON.")

        # Set the chamber temperature dynamically using Invoke
        set_temperature_property = plate_settings.GetType().GetProperty("SetChamberTemperatureCelsius")
        set_temperature_property.SetValue(plate_settings, Single(target_temperature_celsius))
        
        print(f"Chamber temperature set to {target_temperature_celsius}°C.")
        
        # Check the current plate temperature until it reaches the target
        while True:
            # Access the ReadChamberTemperatureCelsius property (which returns an array of float)
            read_temperature_property = plate_settings.GetType().GetProperty("ReadChamberTemperatureCelsius")
            current_temperatures = read_temperature_property.GetValue(plate_settings)
            
            # Check if the temperatures array is not empty
            if current_temperatures is None or len(current_temperatures) == 0:
                print("Error: No temperature readings available!")
                return
            
            # Use the first temperature value (assuming it represents the chamber temperature)
            current_temperature = current_temperatures[0]
            print(f"Current chamber temperature: {current_temperature}°C")
            
            # Check if the temperature has reached or exceeded the target
            if current_temperature >= target_temperature_celsius:
                print(f"Target temperature of {target_temperature_celsius}°C reached.")
                break  # Exit the loop
            
            # Wait for a short time before checking the temperature again
            time.sleep(10)  # Wait 10 seconds before checking again

    except Exception as e:
        print(f"Error setting chamber temperature: {e}")


def terminate_acquisition():
    """
    Terminates acquisition by stopping streaming, releasing BioCam, 
    and closing resources safely.
    """
    global bioCam, is_streaming, slot_index, meaPlatePilot

    if bioCam is None:
        print("No active BioCam to terminate.")
        return

    try:
        # Stop streaming if active
        if is_streaming:
            stop_streaming_method = bioCam.GetType().GetMethod("StopDataStreaming")
            stop_streaming_method.Invoke(bioCam, [])
            is_streaming = False
            print("Data streaming stopped successfully.")

        # Turn off Chamber heater
        
        # Access MeaPlateSettings from IMeaPlatePilot
        settings_property = meaPlatePilot.GetType().GetProperty("Settings")
        plate_settings = settings_property.GetValue(meaPlatePilot)
        
        # Check if temperature control is on
        is_temp_control_on_property = plate_settings.GetType().GetProperty("IsChamberTemperatureControlOn")
        is_temp_control_on = is_temp_control_on_property.GetValue(plate_settings)

        print(f"Chamber temperature control is {'ON' if is_temp_control_on else 'OFF'}.")

        # If it's ON, turn it OFF
        if is_temp_control_on:
             is_temp_control_on_property.SetValue(plate_settings, False)
             print("Chamber temperature control has been turned OFF.")

        # Close the BioCam connection
        close_method = bioCam.GetType().GetMethod("Close")
        close_method.Invoke(bioCam, [True])  # Pass True to indicate closing the board
        print("BioCam connection closed successfully.")

        # Release BioCam control via BioCamPool
        from System import Int32
        slot_index_int32 = Int32(slot_index)  # Convert slot index to System.Int32
        release_control_method = bioCamPoolType.GetMethod("ReleaseBioCamControl")
        release_control_method.Invoke(None, [slot_index_int32])
        print(f"BioCam control released successfully at slot index {slot_index_int32}.")

    except Exception as ex:
        print(f"Error during termination: {ex}")

    finally:
        
        # Ensure all resources are reset
        bioCam = None
        is_streaming = False
        print("All resources have been reset.")

        print("Acquisition terminated safely.")

def save_streamed_data(h5_filename, data_queue):
    """
    Saves streamed MEA data to an HDF5 file asynchronously.
    - Handles variable sampling rates dynamically.
    - Buffers a specified number of chunks before writing to improve performance.
    - Saves both electrode data (4096 electrodes) and timestamps.
    """
    
    global stop_event
    with h5py.File(h5_filename, "w") as h5file:
        # Initialize dataset for 1D vector (4096 * samples) instead of 2D matrix
        dset_data = h5file.create_dataset("electrode_data", 
                                          shape=(0,),  # No initial data, will grow dynamically
                                          maxshape=(None,),  # Number of samples can grow
                                          dtype='uint16')
        dset_timestamps = h5file.create_dataset("timestamps", 
                                                shape=(0,),  # No initial timestamps
                                                maxshape=(None,),  # Number of timestamps can grow
                                                dtype='int64')

        print(f"Saving streamed MEA data to {h5_filename}")

        buffer_data = []
        buffer_timestamps = []
        expected_samples = None  # To determine sampling rate dynamically

        # Set batch size (in chunks)
        batch_size = 10  # Adjust this based on your needs (e.g., write after every 10 chunks)

        while not stop_event.is_set():
            try:
                # Retrieve data and timestamp
                data, timestamp = data_queue.get(timeout=1)

                # Ensure the data is a 1D vector and not empty
                if data.shape[0] == 0:  # Skip empty data chunks
                    continue

                # Set expected number of samples dynamically
                if expected_samples is None:
                    expected_samples = data.shape[0]  # Number of samples in one chunk
                    # print(f"Detected sampling rate: {expected_samples} samples per chunk.")

                # Ensure the sample count is consistent
                if data.shape[0] != expected_samples:
                    # print(f"Warning: Inconsistent data shape {data.shape}, expected {expected_samples} samples.")
                    continue

                # Store in buffer
                buffer_data.append(data)  # Data is already 1D
                buffer_timestamps.append(timestamp)

                # Write in batches (every `batch_size` chunks)
                if len(buffer_data) >= batch_size:
                    num_new_samples = len(buffer_data)

                    # Debugging print statements to check the shapes
                    # print(f"Buffer Data Shape: {np.array(buffer_data).shape}")
                    # print(f"Expected Shape of dset_data: {dset_data.shape}")
                    # print(f"Writing {num_new_samples} new samples")

                    # Resize datasets before writing to match new number of samples
                    # Resize to accommodate the correct number of samples (not just 10)
                    total_new_samples = num_new_samples * expected_samples
                    dset_data.resize(dset_data.shape[0] + total_new_samples, axis=0)  # Resize along the sample dimension
                    dset_timestamps.resize(dset_timestamps.shape[0] + num_new_samples, axis=0)  # Resize timestamps

                    # Debugging print statements to check data sizes
                    # print(f"dset_data shape after resize: {dset_data.shape}")
                    # print(f"Buffer data shape: {np.array(buffer_data).shape}")

                    # Flatten buffer_data to ensure it's a 1D vector
                    flat_data = np.concatenate(buffer_data)

                    # Check if the data length matches
                    if flat_data.shape[0] != dset_data[-total_new_samples:].shape[0]:
                        # print(f"Shape mismatch! Cannot write {flat_data.shape[0]} to {dset_data[-total_new_samples:].shape[0]}")
                        continue
                    
                    # Convert to NumPy arrays for efficiency
                    dset_data[-total_new_samples:] = flat_data  # Write flattened data
                    dset_timestamps[-num_new_samples:] = np.array(buffer_timestamps)  # Write timestamps

                    # Clear buffer after writing
                    buffer_data.clear()
                    buffer_timestamps.clear()

            except queue.Empty:
                continue  # Wait for more data
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

MAX_MEMORY_BYTES = 10 * 1024 * 1024  # 10 MB
SAVE_DIR = "streamed_data"
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
    """
    Set up data streaming, register the event handler, and handle duplicate timestamps.
    """
    global bioCam, is_streaming, first_chunk_saved, file_counter

    # Initialize streaming state
    is_streaming = False

    try:
        # Add the event handler for data reception
        def on_data_received(sender, e):
            """
            Callback function triggered when data is received.
            Stores raw data and timestamps in the queue and checks for duplicates.
            """
            global first_chunk_saved, file_counter
            try:
                # Start timing the callback
                callback_start = time.time()


                # Convert e.payload into a numpy array of uint16
                data = np.frombuffer(e.Payload, dtype=np.uint16)
                # Convert payload to a numpy array
                # data = np.array(e.Payload, dtype = np.uint16)
                
                # Check the length and structure
                length_of_data = len(data)  # Should now be nframes * nchannels
                print(f"Length of data: {length_of_data}")
                print(f"Data type: {data.dtype}")

         

                if len(data) == 0:
                    print("Warning: Empty data payload received!")
                    return

                # Get the timestamp from the data header
                timestamp = e.Header.Timestamp
                # # type of the timestamps
                # print(f"Received timestamps type: {type(timestamp)}")
                # print(f"timestamps shape: {timestamp.shape}")
                # print(f"data chunk status: {first_chunk_saved}")

                # Check for duplicate timestamps
                if not data_queue.empty():
                    _, latest_timestamp = data_queue.queue[-1]  # Peek at the last item in the queue
                    if timestamp == latest_timestamp:
                        print(f"Duplicate timestamp detected: {timestamp}, Payload size={len(data)}")
                        return  # Skip adding this duplicate to the queue


                # Manage the data queue
                if data_queue.qsize() >= max_queue_size:
                    data_queue.get()  # Remove the oldest chunk
                data_queue.put((data, timestamp))  # Add the new chunk

                # Manage the timestamp queue
                if timestamp_queue.qsize() >= max_queue_size:
                    timestamp_queue.get()  # Remove the oldest timestamp
                timestamp_queue.put(timestamp)  # Add the new timestamp
                """
                # Increment the file counter for unique filenames
                
                file_counter += 1
                file_name = f"data_chunk_{file_counter}.mat"  # Use the counter in the filename
               

                # Create a dictionary with data and timestamp
                data_dict = {
                    'data': data,  # Payload
                    'timestamp': timestamp  # Timestamp
                    }
                    
                # Debugging print to confirm the data is being saved
                print(f"Saving data chunk to {file_name}...")
                


                # Save the dictionary to a .mat file
                scipy.io.savemat(file_name, data_dict)

                # Set the flag to prevent overwriting
                """

                # Log execution time (optional, can be removed for performance)
                callback_end = time.time()
                #print(f"on_data_received started at {callback_start-program_start}, executed in {callback_end - callback_start:.3f} seconds, timestamps = {timestamp}")

            except Exception as ex:
                print(f"Error in on_data_received: {ex}")

        # Register the data reception event
        data_received_event = bioCam.GetType().GetEvent("DataReceived")
        from System import EventHandler
        data_received_delegate = EventHandler[dataPacketReceivedEventArgsType](on_data_received)
        data_received_event.AddEventHandler(bioCam, data_received_delegate)

        # Start streaming data (only once!)
        start_streaming_method = bioCam.GetType().GetMethod("StartDataStreaming")
        start_streaming_method.Invoke(bioCam, [Int32(100), False])  # 10 ms chunks
        is_streaming = True
        print("Data streaming started successfully.")

    except Exception as e:
        print(f"Error during BioCAM initialization: {e}")

def get_mea_chunk_data(raw_data, timestamp):
    """
    Processes raw MEA data: reshapes and applies filtering.

    Parameters:
    - raw_data: 1D numpy array of raw MEA data.
    - timestamp: Timestamp corresponding to the data chunk.

    Returns:
    - filtered_data: 2D numpy array (filtered MEA data).
    - timestamp: Same timestamp as input, returned for consistency.
    """
    if raw_data is None or timestamp is None:
        return None, None

    # Reshape the raw data
    num_channels = 4096
    n_timestamps = len(raw_data) // num_channels
    reshaped_data = np.reshape(raw_data, (num_channels, n_timestamps))

    # Apply high-pass filter to the reshaped data
    #filtered_data = highpass_filter(reshaped_data)

    return reshaped_data, timestamp


def detect_spikes(filtered_data, std_threshold):
    """
    Detect spikes from the filtered_data (4096 x T), calculate activity in M1 and M2 regions,
    and return spike counts along with a movement decision.

    Parameters:
    - filtered_data: 2D numpy array of shape (4096, T), the filtered MEA data.
    - M1, M2: Indices of the motor regions (predefined).
    - std_threshold: Threshold for spike detection (default is 8 times the standard deviation).

    Returns:
    - decision: 'left', 'right', or 'none' based on M1 and M2 activity.
    - spike_counts: 1D numpy array of spike counts for all electrodes.
    """
    # Initialize spike counts

    global M1, M2
    spike_counts = np.zeros(4096)

    activity_M1 = 0
    activity_M2 = 0

    # Combine M1 and M2 indices
    target_indices = np.concatenate([M1, M2])

    # Iterate over motor electrodes
    for i in target_indices:
        signal = filtered_data[i, :]  # Signal for the i-th electrode
        mean = np.mean(signal)
        std = np.std(signal)

        # Define thresholds
        positive_threshold = mean + std_threshold * std
        negative_threshold = mean - std_threshold * std

        # Count spikes above and below thresholds
        spikes = np.sum(signal > positive_threshold) + np.sum(signal < negative_threshold)
        spike_counts[i] = spikes

        # Add spike counts to M1 and M2 activity if the electrode belongs to the respective region
        if i in M1:
            activity_M1 += spikes
        elif i in M2:
            activity_M2 += spikes

    # Determine decision based on M1 and M2 activities
    if activity_M1 > activity_M2:
        decision = "left"
    elif activity_M2 > activity_M1:
        decision = "right"
    else:
        decision = "none"

    return decision, spike_counts



# --- Ball Update Logic ---


def random_velocity():
    """
    Generate a random initial velocity with a normalized magnitude.
    """
    angle = np.random.rand() * 2 * np.pi  # Random angle in radians
    speed = 0.1  # Set the speed (adjust as needed)
    return [speed * np.cos(angle), speed * np.sin(angle)]

#def update_ball(ball_position, ball_velocity, paddle_position):
#    """
#    Update ball position and velocity based on collisions and movement.
    
#    Parameters:
#        ball_position (list): [x, y] coordinates of the ball.
#        ball_velocity (list): [vx, vy] velocity of the ball.
#        paddle_position (list): [x, y] coordinates of the paddle center.
#        paddle_width (float): Width of the paddle.
        
#    Returns:
#        ball_position (list): Updated [x, y] coordinates of the ball.
#        ball_velocity (list): Updated [vx, vy] velocity of the ball.
#        flag (int): Status flag indicating game state.
#    """
#    global FPS  # Use the global FPS variable
#    dt = 1 / FPS  # Calculate the time step
#    paddle_width = 0.2

#    # Update ball position based on velocity
#    ball_position = [ball_position[0] + ball_velocity[0] * dt,
#                     ball_position[1] + ball_velocity[1] * dt]

#    # Ball bouncing off top and bottom walls
#    if ball_position[1] <= 0:
#        # Ball hits the bottom wall (miss)
#        flag = 1  # Set flag to 2 for a miss
#        ball_position = [0.5, 0.5]  # Reset ball to the center
#        ball_velocity = random_velocity()  # Assign a random initial velocity
#        return ball_position, ball_velocity, flag
#    elif ball_position[1] >= 1:
#        ball_velocity[1] = -abs(ball_velocity[1])  # Reverse Y velocity downwards
#        ball_position[1] = 1  # Correct position to stay in bounds

#    # Ball bouncing off left and right walls
#    if ball_position[0] <= 0:
#        ball_velocity[0] = abs(ball_velocity[0])  # Reverse X velocity to the right
#        ball_position[0] = 0  # Correct position to stay in bounds
#    elif ball_position[0] >= 1:
#        ball_velocity[0] = -abs(ball_velocity[0])  # Reverse X velocity to the left
#        ball_position[0] = 1  # Correct position to stay in bounds

#    # Ball bouncing off the paddle
#    paddle_top = paddle_position[1] + 0.05  # Paddle's height boundary
#    paddle_bottom = paddle_position[1] - 0.05
#    paddle_left = paddle_position[0] - paddle_width / 2
#    paddle_right = paddle_position[0] + paddle_width / 2

#    if (paddle_bottom <= ball_position[1] <= paddle_top and
#        paddle_left <= ball_position[0] <= paddle_right):
#        flag = 1  # Set flag to 3 for a hit
#        ball_velocity[1] = abs(ball_velocity[1])  # Bounce upward
#        ball_position[1] = paddle_top  # Place the ball just above the paddle
#        return ball_position, ball_velocity, flag

#    # Ball is in play but hasn't hit anything critical
#    flag = 1  # Set flag to 1 for normal movement
#    return ball_position, ball_velocity, flag

def update_ball(ball_position, ball_velocity, paddle_position):
    """
    Update ball position and velocity based on collisions and movement.

    Parameters:
        ball_position (list): [x, y] coordinates of the ball.
        ball_velocity (list): [vx, vy] velocity of the ball.
        paddle_position (list): [x, y] coordinates of the paddle center.

    Returns:
        ball_position (list): Updated [x, y] coordinates of the ball.
        ball_velocity (list): Updated [vx, vy] velocity of the ball.
        paddle_position (list): Paddle position (unchanged unless reset).
        flag (int): Status flag indicating game state.
        Game ongoing: flag = 1
        Hit : flag = 3 (Reward)
        Miss : flag = 2 (Punishment)
    """
    global FPS  # Use the global FPS variable
    dt = 1 / FPS  # Calculate the time step
    paddle_width = 0.33

    # Update ball position based on velocity
    ball_position = [ball_position[0] + ball_velocity[0] * dt,
                     ball_position[1] + ball_velocity[1] * dt]

    # Ball passing behind the paddle (miss: hits the top wall)
    if ball_position[1] >= 1:
        # Reset the game when the ball misses the paddle
        flag = 2  # Miss = Punishment condition
        ball_position = [0.5, 0.5]  # Reset ball to the center
        ball_velocity = random_velocity()  # Assign a random initial velocity
        return ball_position, ball_velocity, flag

    # Ball hitting the bottom wall
    if ball_position[1] <= 0:
        ball_velocity[1] = abs(ball_velocity[1])  # Reflect Y velocity upward
        ball_position[1] = 0  # Correct position to stay in bounds

    # Ball hitting the left wall
    if ball_position[0] <= 0:
        ball_velocity[0] = abs(ball_velocity[0])  # Reflect X velocity to the right
        ball_position[0] = 0  # Correct position to stay in bounds

    # Ball hitting the right wall
    if ball_position[0] >= 1:
        ball_velocity[0] = -abs(ball_velocity[0])  # Reflect X velocity to the left
        ball_position[0] = 1  # Correct position to stay in bounds

    # Ball hitting the paddle
    paddle_top = paddle_position[1] + 0.05  # Paddle's height boundary
    paddle_bottom = paddle_position[1] - 0.05
    paddle_left = paddle_position[0] - paddle_width / 2
    paddle_right = paddle_position[0] + paddle_width / 2

    if (paddle_bottom <= ball_position[1] <= paddle_top and
        paddle_left <= ball_position[0] <= paddle_right):
        flag = 3  # Hit = Reward condition
        ball_velocity[1] = -abs(ball_velocity[1])  # Reflect Y velocity downward
        ball_position[1] = paddle_bottom  # Place the ball just below the paddle
        return ball_position, ball_velocity, flag

    # Ball is in play but hasn't hit anything critical
    flag = 1  # Normal play
    return ball_position, ball_velocity, flag





# -------------------------------- stimulation -------------------------------------------------------
# Stimulation related classes that are called only once in the main 

# Create the ChCoord instances for all electrodes
def create_chcoords(electrodes):
    ch_coords = []
    constructor_2 = chCoordType.GetConstructor([Int32, Int32])  # 2-parameter constructor
    for el in electrodes:
        param1 = Int32(el[0])
        param2 = Int32(el[1])
        ch_coord_instance = constructor_2.Invoke([param1, param2])
        ch_coords.append(ch_coord_instance)
    return ch_coords

# Define the function to create endpoints instances
def create_endpoints(chcoord_list, label_prefix):
    """
    Create a list of StimEndPointDupleX instances from a list of ChCoord objects.
    
    Parameters:
    - chcoord_list: list of ChCoord instances
    - StimEndPointDupleXType: the reflected type for StimEndPointDupleX class
    - label_prefix: a string label to help name the endpoints (e.g., "Pos" or "Neg")
    
    Returns:
    - A list of StimEndPointDupleX instances.
    """
    endpoints = []
    endpoint_constructor = StimEndPointDupleXType.GetConstructor([
        chCoordType, String, String, String
    ])

    for idx, ch in enumerate(chcoord_list):
        name = String(f"{label_prefix}_EP_{idx}")
        location = String(f"Index_{idx}")  # You can modify this if needed
        description = String("Auto-generated endpoint")

        endpoint = endpoint_constructor.Invoke([ch, name, location, description])
        endpoints.append(endpoint)

    return endpoints

def create_stimulation_protocols(positive_endpoints,negative_endpoints):
   


    # Step 1: Create an instance of the StimProperties class
    time_resolution = Int32(time_resolution_value)               # Resolution of time in microseconds
    amplitude_resolution = Double(amplitude_resolution_value)    # Resolution of amplitude in microAmps
    min_amplitude = Int32(-25)                                   # Minimum amplitude in microAmps         
    max_amplitude = Int32(25)                                    # Maximum amplitude in microAmps

    # Define the 4-parameter constructor for StimProperties class
    constructor_4 = StimPropertiesType.GetConstructor([
        Int32, System.Double, Int32, Int32])  # 4 parameters constructor

    # Invoke the constructor to create a StimProperties instance
    try:
        stim_properties = constructor_4.Invoke([
            time_resolution, amplitude_resolution, min_amplitude, max_amplitude
        ])
    except Exception as e:
        print(f"Error setting StimProperties: {e}")

    # Step 2: Define the constructor for RectangularStimPulse once outside the loop
    constructor_7 = rectangularStimPulseType.GetConstructor([
        String, StimPropertiesType, System.Double, Int32, Int32, System.Double, Int32
    ])  # 7 parameters constructor

    # Step 3: Loop over pulse types and create the pulses

    pulses = []  # Initialize a list to store all the pulses

    for i in range(len(pulse_types)):
        name = String(pulse_types[i])
        amp1 = Double(pulse_amplitude[i][0])  # Amplitude for the first phase
        amp2 = Double(pulse_amplitude[i][1])  # Amplitude for the second phase
        width1 = Int32(W1)                    # Width for the first phase
        inter_width = Int32(IW)               # Inter-phase width
        width2 = Int32(W2)                    # Width for the second phase

        # Invoke the constructor to create the pulse instance
        try:
            pulse = constructor_7.Invoke([
                name,                    # Friendly name for the pulse
                stim_properties,          # StimProperties object
                amp1,                     # Amplitude for first phase
                width1,                   # Width for the first phase
                inter_width,              # Inter-phase width
                amp2,                     # Amplitude for the second phase
                width2                    # Width for the second phase
            ])

            pulses.append(pulse)  # Store the pulse in the list
            print(f"Created pulse {name}")
        except Exception as e:
            print(f"Failed to create RectangularStimPulse instance for {name}: {e}")

    # The first pulse in the list is the SensoryPulse
    sensory_pulse = pulses[0]

    # Step 4: Generate 37 sensory protocols based on frequency
    protocols = []  # List to store the protocol instances
    for freq in range(4, 41):  # Frequency range from 4 Hz to 40 Hz
        protocol_name = String(f"SensoryProtocol_{freq}Hz")
        pulse_rate = Double(freq)  # Frequency in Hz
        count = Int32(freq)  # Number of pulses based on frequency (e.g., 100 pulses for 100Hz)

        # Define the constructor for StimTrainProtocol
        constructor_5 = StimTrainProtocolType.GetConstructor([
            String, rectangularStimPulseType, StimPropertiesType, Int32, System.Double
        ])  # 5 parameters constructor

        # Invoke the constructor to create the protocol instance
        try:
            protocol = constructor_5.Invoke([
                protocol_name,  # Protocol name
                sensory_pulse,  # Use the first pulse (SensoryPulse)
                stim_properties,  # StimProperties object
                count,  # Number of pulses based on frequency
                pulse_rate  # Pulse rate in Hz
            ])
            

            
            # Assign positive and negative endpoints to the protocol
            positive_prop = protocol.GetType().GetProperty("PositiveEndPoints")
            positive_prop.SetValue(protocol, positive_endpoints)

            negative_prop = protocol.GetType().GetProperty("NegativeEndPoints")
            negative_prop.SetValue(protocol, negative_endpoints)

            protocols.append(protocol)  # Store the protocol in the list
            print(f"Created protocol {protocol_name} with {freq}Hz frequency")
        except Exception as e:
            print(f"Failed to create Sensory Protocol at {freq}Hz: {e}")

    # Step 5: Generate 2 reward protocols w/ and w/o delay

    # Define parameters for the first protocol (5 pulses with no delay)
    protocol_name_1 = String("BurstProtocol_NoDelay")
    reward_pulse = pulses[1]            # Use the 2nd pulse (RewardPulse)
    delay_1 = Int32(0)                  # No delay at the beginning (usec)
    count = Int32(5)                    # Number of pulses in the burst (5 pulses)
    distance = Int32(10000)             # Duration of each pulse (usec) (10 ms for 100 Hz)

    constructor_7 = StimTrainProtocolType.GetConstructor([
        String, rectangularStimPulseType, StimPropertiesType,DataSamplingTimeConverterType, Int32, Int32, Int32
    ]) 
    
    default_converter = DataSamplingTimeConverterType.GetConstructor([]).Invoke([])

    # Create the first protocol (5 pulses, no delay)
    protocol_1 = constructor_7.Invoke([
        protocol_name_1,        # Protocol name
        reward_pulse,           # Use the 2nd pulse (reward pulse)
        stim_properties,        # StimProperties object
        default_converter,      # default converter instance 1
        delay_1,                # No delay at the beginning (usec)
        count,                  # Number of pulses in the burst
        distance                # Duration of each pulse (usec)
    ])


    # Assign positive and negative endpoints to the protocol
    positive_prop = protocol_1.GetType().GetProperty("PositiveEndPoints")
    positive_prop.SetValue(protocol_1, positive_endpoints)

    negative_prop = protocol_1.GetType().GetProperty("NegativeEndPoints")
    negative_prop.SetValue(protocol_1, negative_endpoints)

    # Define parameters for the second protocol (5 pulses with 150 ms delay)
    protocol_name_2 = String("BurstProtocol_WithDelay")
    delay_2 = Int32(150000)               # Delay of 150 ms at the beginning (time between bursts)
    # Second protocol will be the same but with a delay at the start
    protocol_2 = constructor_7.Invoke([
        protocol_name_2,        # Protocol name
        reward_pulse,           # Use the 2nd pulse (reward pulse)
        stim_properties,        # StimProperties object
        default_converter,      # default converter instance 1
        delay_2,                # 150 ms delay at the beginning (usec)
        count,                  # Number of pulses in the burst
        distance                # Duration of each pulse (usec)
    ])

    # Assign positive and negative endpoints to the protocol
    positive_prop = protocol_2.GetType().GetProperty("PositiveEndPoints")
    positive_prop.SetValue(protocol_2, positive_endpoints)

    negative_prop = protocol_2.GetType().GetProperty("NegativeEndPoints")
    negative_prop.SetValue(protocol_2, negative_endpoints)

    # Store the protocol in the list
    protocols.append(protocol_1)         # number 38
    protocols.append(protocol_2)         # number 39

    return protocols, pulses



# ------------------------------- stimulation funcs -----------------------------------------------------
# Function to update the stimulation state
def set_stim_state(new_state):
    with stim_state["lock"]:
        stim_state["state"] = new_state
        print(f"Stim state updated to {new_state}")

# Function to get the stimulation state 
def get_stim_state():
    with stim_state["lock"]:
        return stim_state["state"]

def encode_pong_state(ball_position, paddle_position):
    """
    Encodes the pong game state into stimulation parameters and returns
    the electrode indices and frequency for sensory stimulation.
    """
    
    
    # Calculate 2D distance between ball and paddle
    distance = math.sqrt((ball_position[0] - paddle_position[0])**2 + (ball_position[1] - paddle_position[1])**2)
    max_distance = math.sqrt(2)  # Adjust based on your game field dimensions

    # Calculate frequency using the same formula as in Cortical labs paper
    freq = round(4 + (36 * max(0, 1 - distance / max_distance)))  # Frequency encoding with rounding
    
    # Choose electrodes based on the ball position
    left_positive_elecs = [el for el in positive_electrodes if el[1] <= 32]
    right_positive_elecs = [el for el in positive_electrodes if el[1] > 32]

    
    # Select electrodes based on ball's horizontal position relative to paddle
    if ball_position[0] > paddle_position[0]:
        # Ball on the right side, use right electrodes
        selected_positive_elecs = right_positive_elecs
    else:
        # Ball on the left side, use left electrodes
        selected_positive_elecs = left_positive_elecs

    # Get the indices for the selected electrodes
    positive_elec_indices = [positive_electrodes.index(el) for el in selected_positive_elecs]

    stim_params = {
        "frequency": freq,  # Hz (varies between 4-40 Hz for sensory input)
    }

    return stim_params, positive_elec_indices

def stop_and_reset_protocol():
    """
    Stops the currently playing protocol (at index 0) and resets the stimulator.
    Since only one protocol can be loaded at a time, index is always 0.
    """
    global protocol_manager
    try:
        # Check if the protocol is currently playing
        get_protocol_status_method = protocol_manager.GetType().GetMethod("GetProtocolStatus")
        protocol_status = get_protocol_status_method.Invoke(protocol_manager, [Int32(0)])  # Get status for protocol at index 0

        if protocol_status == 4:  # If the status is "Playing" (value 4)
            print("Protocol is currently playing. Stopping protocol...")

            # Stop the protocol at index 0
            stop_protocol_method = protocol_manager.GetType().GetMethod("StopProtocol")
            stop_success = stop_protocol_method.Invoke(protocol_manager, [Int32(0)])

            if stop_success:
                print("Protocol stopped successfully.")
            else:
                print("Failed to stop protocol.")

            # Reset the protocol at index 0
            reset_protocol_method = protocol_manager.GetType().GetMethod("ResetProtocol")
            reset_protocol_method.Invoke(protocol_manager, [Int32(0)])
            print("Protocol reset successfully.")
        else:
            print(f"No protocol is currently playing. Current status: {protocol_status}")

    except Exception as e:
        print(f"Error stopping or resetting protocol: {e}")


def prepare_sensory_protocol(protocols, stim_params, electrode_indices):
    """
    Selects the sensory protocol based on frequency, modifies the endpoints according to the selected electrodes,
    and returns the updated protocol ready for loading and playing.

    Parameters:
    - protocols: List of all available protocols (should be 37 sensory + 2 reward/punishment initially).
    - freq: Frequency selected from encode_pong_state output.
    - electrode_indices: List of indices (e.g., [0, 3, 5]) corresponding to selected electrodes.
    - positive_endpoints_full: Full list of positive endpoints (pre-generated once).
    - negative_endpoints_full: Full list of negative endpoints (pre-generated once).

    Returns:
    - protocol: The modified protocol ready to load/play.
    """
    try:
        # Calculate the protocol index
        frequency_hz = stim_params["frequency"]
        protocol_index = frequency_hz - 4  # because protocols start from 4Hz

        if protocol_index < 0 or protocol_index >= len(protocols):
            raise ValueError(f"Invalid frequency {freq}: no matching protocol in protocols list!")

        # Fetch the protocol
        protocol = protocols[protocol_index]

        # Select the endpoints corresponding to electrode_indices
        selected_positive_endpoints = [positive_endpoints_full[i] for i in electrode_indices]
        selected_negative_endpoints = [negative_endpoints_full[i] for i in electrode_indices]

        # Access properties through reflection
        positive_prop = protocol.GetType().GetProperty("PositiveEndPoints")
        negative_prop = protocol.GetType().GetProperty("NegativeEndPoints")

        # Set the selected endpoints
        positive_prop.SetValue(protocol, selected_positive_endpoints)
        negative_prop.SetValue(protocol, selected_negative_endpoints)

        print(f"Prepared protocol at frequency {freq} Hz with selected electrodes {electrode_indices}.")

        return protocol

    except Exception as e:
        print(f"Error preparing sensory protocol: {e}")
        return None

def load_and_start_protocol(protocol, protocol_type):
    """
    Load a protocol from the protocol list, check if it's ready, and then start it.
    After the protocol finishes, set stim_state to "none" (for reward and punishment).
    
    Parameters:
    - protocol: A protocol from the list of all protocols (including sensory and reward).
    - protocol_type: The type of protocol.
    """
    global protocol_manager
    try:

        # Load the protocol into the stimulator (since only one protocol can be loaded at a time)
        load_protocol_method = protocol_manager.GetType().GetMethod("LoadProtocol")
        load_success = load_protocol_method.Invoke(protocol_manager, [Int32(0), protocol])  # Always use index 0

        if load_success:  # Check if the protocol is successfully loaded
            print(f"{protocol_type.capitalize()} protocol successfully loaded.")
            
            # Start the protocol directly once loaded
            start_protocol_method = protocol_manager.GetType().GetMethod("StartProtocol")
            start_success = start_protocol_method.Invoke(protocol_manager, [Int32(0)])  # Start protocol at index 0
            
            if start_success:
                print(f"{protocol_type.capitalize()} protocol started successfully.")
                
            else:
                print(f"Failed to start {protocol_type} protocol.")
        else:
            print(f"Failed to load {protocol_type} protocol.")
    
    except Exception as e:
        print(f"Error in loading or starting {protocol_type} protocol: {e}")


def load_and_start_theta_burst(protocol, protocol_type, repetitions=5, burst_duration=0.05, delay_between_bursts=0.15):
    """
    Load a protocol into the stimulator and apply theta-burst stimulation.
    The protocol (e.g., reward) is applied multiple times with a delay between bursts.

    Parameters:
    - protocol: The protocol object to load (from your protocol list).
    - protocol_type: A string, like "reward" or "punishment", for printing.
    - repetitions: How many bursts to apply (default 5).
    - burst_duration: Duration of one burst (default 50 ms for 100 Hz, 5 pulses).
    - delay_between_bursts: Delay between bursts (default 150 ms).
    """
    global protocol_manager
    try:
        # Load the protocol into the stimulator (always at index 0)
        load_protocol_method = protocol_manager.GetType().GetMethod("LoadProtocol")
        load_success = load_protocol_method.Invoke(protocol_manager, [Int32(0), protocol])

        if load_success:
            print(f"{protocol_type.capitalize()} protocol loaded successfully.")

            # Now apply theta-burst stimulation
            start_protocol_method = protocol_manager.GetType().GetMethod("StartProtocol")
            
            for burst_num in range(repetitions):
                print(f"Starting burst {burst_num + 1}/{repetitions} for {protocol_type} stimulation")
                
                # Start the protocol (burst)
                start_success = start_protocol_method.Invoke(protocol_manager, [Int32(0)])
                
                if not start_success:
                    print(f"Failed to start {protocol_type} burst {burst_num + 1}")
                    break  # Exit if we can't start
                
                # Wait for the burst duration (stimulation happening)
                time.sleep(burst_duration)

                if burst_num < repetitions - 1:
                    # Wait between bursts (delay)
                    time.sleep(delay_between_bursts)

            print(f"Theta-burst {protocol_type} stimulation completed.")

        else:
            print(f"Failed to load {protocol_type} protocol.")

    except Exception as e:
        print(f"Error in loading/starting {protocol_type} theta-burst protocol: {e}")

def apply_punishment_stimulation(stimulator, positive_endpoints_full,negative_endpoints_full, pulse, duration=4, rest = 4):
    """
    Apply punishment stimulation with 5 Hz frequency for 4 seconds, following by 4 sec rest
    
    Parameters:
    - stimulator: The BioCAM stimulator object.
    - available_electrodes: List of available electrodes to choose from.
    - pulse: The predefined RectangularStimPulse object.
    - duration: Total duration for punishment stimulation (default 4 sec).
    """
    freq = 5
    total_pulses = int(duration * freq)  # 5 Hz -> 5 pulses per second -> 20 pulses in 4 sec
    pulse_interval_sec = 0.2  # 200 ms = 0.2 sec between each pulse

    # Define the pulse duration (200 + 25 + 200) microseconds
    pulse_duration = 200 + 25 + 200  # 425 μs
    
    # Preselect the first random set of electrodes
    num_electrodes = random.randint(1, 8)                       # Choose between 1 and 8 electrodes
    selected_indices = random.sample(range(8), num_electrodes)  # Randomly select 'num_electrodes' indices between 0 and 7
    
    # Select the endpoints corresponding to the selected indices
    selected_positive_endpoints = [positive_endpoints_full[i] for i in selected_indices]
    selected_negative_endpoints = [negative_endpoints_full[i] for i in selected_indices]
    
    send_method = stimulator.GetType().GetMethod("Send")  # Get the Send method

    print(f"Starting punishment stimulation: {total_pulses} pulses at 5 Hz")

    for i in range(total_pulses):
        # Send the stimulation pulse **first** before selecting next electrodes
          
        try:
            send_method.Invoke(stimulator, [pulse, selected_positive_endpoints, selected_negative_endpoints, Double(0)])  # Timestamp = 0 for immediate execution
            print(f"Pulse {i+1}/{total_pulses}: Stimulated {num_electrodes} electrodes: {selected_electrodes}")
        except Exception as e:
            print(f"Error sending stimulation pulse {i+1}: {e}")
        
        # Start timer **after** sending the pulse, to measure electrode selection time
        start_time = time.time()
        
        # Select new electrodes for the next pulse (this happens while waiting)
        num_electrodes = random.randint(1, 8)                       # Choose between 1 and 8 electrodes
        selected_indices = random.sample(range(8), num_electrodes)  # Randomly select 'num_electrodes' indices between 0 and 7
    
        # Select the endpoints corresponding to the selected indices
        selected_positive_endpoints = [positive_endpoints_full[j] for j in selected_indices]
        selected_negative_endpoints = [negative_endpoints_full[j] for j in selected_indices]
        
        # Get the current time spent on selecting electrodes (in microseconds)
        selection_time = (time.time() - start_time) * 1000000  # Convert to microseconds
        
        # Calculate the remaining time to sleep dynamically
        remaining_time = 200000 - pulse_duration - selection_time  # 200 ms total, minus used time
        sleep_time = max(remaining_time / 1000000, 0)  # Convert to seconds, ensure non-negative

        # Sleep for the remaining time
        time.sleep(sleep_time)

    print("Punishment stimulation completed and 4 sec rest started")
    time.sleep(rest)






    
# --- Visualization Functions ---
def plot_pong_game(ball_position, paddle_position, ax):
    """
    Visualizes the Pong game in real-time.
    """
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Draw the ball
    ball = plt.Circle(ball_position, 0.02, color='red')
    ax.add_artist(ball)

    # Draw the paddle
    paddle_x = paddle_position[0]
    paddle_width = 0.33
    paddle_y = 0.9
    paddle = plt.Rectangle((paddle_x - paddle_width / 2, paddle_y), paddle_width, 0.02, color='blue')
    ax.add_artist(paddle)

    # Draw walls
    ax.plot([0, 1], [1, 1], color='black')  # Top wall
    ax.plot([0, 0], [0, 1], color='black')  # Left wall
    ax.plot([1, 1], [0, 1], color='black')  # Right wall
    ax.plot([0, 1], [0, 0], color='black')  # Bottom wall


    ax.set_title("Pong Game")
    ax.axis('off')


# def plot_mea_heatmap(ax,spike_counts,colorbar=None):
#     """
#     Visualizes the MEA activity as a heatmap using global spike_counts.
#     """

#     if spike_counts is None:
#         print("Spike counts are not yet computed.")
#         return colorbar 

#     # Reshape spike_counts into a 64x64 matrix
#     spike_activity_matrix = spike_counts.reshape((64, 64))

#     # Plot the heatmap
#     ax.clear()
#     # Plot the heatmap
#     heatmap = ax.imshow(spike_activity_matrix, cmap='hot', interpolation='nearest', origin='lower')

#     # Set aspect ratio to equal
#     ax.set_aspect('equal')

#     # Remove any previously attached colorbars, but only if valid
#     if colorbar is not None and isinstance(colorbar, Colorbar):
#         try:
#             colorbar.remove()  # Remove the existing colorbar safely
#         except Exception as e:
#             print(f"Failed to remove colorbar: {e}")
#     # Add a new colorbar
#     colorbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
#     ax.set_title("MEA Spike Activity Heatmap")
#     ax.axis('off')

#     return colorbar


def plot_mea_heatmap(ax, spike_counts, heatmap=None, colorbar=None):
    """
    Visualizes the MEA activity as a heatmap using spike_counts.

    Parameters:
    - ax: The matplotlib axis to plot on.
    - spike_counts: The spike counts to visualize.
    - heatmap: The existing heatmap object to update (if any).
    - colorbar: The existing colorbar object to update (if any).

    Returns:
    - heatmap: The updated or newly created heatmap.
    - colorbar: The updated or newly created colorbar.
    """
    if spike_counts is None:
        print("Spike counts are not yet computed.")
        return heatmap, colorbar

    # Reshape spike_counts into a 64x64 matrix
    spike_activity_matrix = spike_counts.reshape((64, 64))

    if heatmap is None:
        # Create the heatmap if it doesn't exist
        heatmap = ax.imshow(spike_activity_matrix, cmap='hot', interpolation='nearest', origin='lower')
        ax.set_title("MEA Spike Activity Heatmap")
        ax.axis('off')
    else:
        # Update the heatmap data if it already exists
        heatmap.set_data(spike_activity_matrix)

    # Update the colorbar
    if colorbar is None:
        colorbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    else:
        # Update the colorbar with the new heatmap
        colorbar.mappable.set_clim(vmin=spike_activity_matrix.min(), vmax=spike_activity_matrix.max())
        colorbar.update_normal(heatmap)

    # Scale fake ticks based on the real caxis range (0 to 700)
    real_min = spike_activity_matrix.min()
    real_max = spike_activity_matrix.max()
   
    # Define fake ticks (from 0 to 20 with pitch of 2)
    fake_ticks = np.arange(0, 21, 2)  # Fake numbers between 0 and 20
   
    # Scale these fake ticks to the real color range
    scaled_fake_ticks = real_min + (fake_ticks / 20) * (real_max - real_min)

    # Create corresponding labels for the fake ticks
    fake_tick_labels = [str(i) for i in fake_ticks]

    # Apply custom scaled ticks and labels to the colorbar
    colorbar.set_ticks(scaled_fake_ticks)
    colorbar.set_ticklabels(fake_tick_labels)

    

    return heatmap, colorbar


# --- Visualization Thread ---


def visualization_thread(mea_to_viz_queue, pong_to_viz_queue):
    """
    Thread to handle updating data for real-time visualization of Pong and MEA activity.
    """
    plt.ion()  # Enable interactive mode
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    pong_ax, mea_ax = axes

    # Initialize default values for ball/paddle positions and spike counts
    ball_position = [0.5, 0.5]
    paddle_position = [0.5, 0.9]
    spike_counts = np.zeros(4096)  # Default empty spike counts
    colorbar = None  # For managing the MEA heatmap colorbar
    heatmap = None  # For the MEA heatmap

    try:
        while not stop_event.is_set():
            # Check for updates from the Pong thread
            try:
                pong_data = pong_to_viz_queue.get(timeout=0.1)
                ball_position = pong_data.get("ball_position", ball_position)
                paddle_position = pong_data.get("paddle_position", paddle_position)
            except queue.Empty:
                # No update from Pong thread; retain last known positions
                pass

            # Check for updates from the MEA thread
            try:
                spike_counts, timestamp = mea_to_viz_queue.get(timeout=0.1)
            except queue.Empty:
                # No update from MEA thread; retain last known spike counts
                pass

            # Debug: Print the visualization update
            print(f"Visualization Update: Ball={ball_position}, Paddle={paddle_position}")

            # Update plots
            plot_pong_game(ball_position, paddle_position, pong_ax)
            heatmap, colorbar = plot_mea_heatmap(mea_ax, spike_counts, heatmap=heatmap, colorbar=colorbar)
            plt.draw()  # Update the plot
            plt.gcf().canvas.flush_events()  # Allow events to process

    finally:
        plt.close(fig)

        


# --- Stimulation Controller Thread ---
def stimulation_controller(stimulator, pong_to_stim_queue, timestamp_queue, protocols, pulses):
    """
    Controls stimulation logic based on game events.
    Implements gating logic between sensory, reward, and punishment based on stim_state.
    """
    global protocol_manager

    # --- Event handler for ProtocolStatusChanged (this is registered only once) ---
    def on_protocol_status_changed(sender, e):
        """
        Event handler for ProtocolStatusChanged event.
        This will check the status of the protocol and set the stim_state to 'none' when finished.
        """
        if e.Status == 5:  # Status '5' means the protocol is finished (StimStatus.Finished)
            print(f"Protocol {e.Index} finished.")
            set_stim_state("none", True)  # Set stim_state to 'none' after the protocol finishes

    # Register the ProtocolStatusChanged event only once (this is done once when the controller starts)
    protocol_status_changed_event = protocol_manager.GetType().GetEvent("ProtocolStatusChanged")
    from System import EventHandler
    protocol_status_delegate = EventHandler[ProtocolStatusChangedEventArgsType](on_protocol_status_changed)
    protocol_status_changed_event.AddEventHandler(protocol_manager, protocol_status_delegate)

    # --- Main game loop ---
    while not stop_event.is_set():
        try:
            # Get game state from Pong
            try:
                game_state_data = pong_to_stim_queue.get(timeout=0.1)
                flag = game_state_data.get("type")  # sensory, reward, or punishment
                ball_position = game_state_data["ball_position"]
                paddle_position = game_state_data["paddle_position"]
                print(f"Received game state: Ball={ball_position}, Paddle={paddle_position}, Flag={flag}")
            except queue.Empty:
                continue

            # Get timestamp (latest available)
            latest_timestamp = None
            while not timestamp_queue.empty():
                latest_timestamp = timestamp_queue.get_nowait()

            if latest_timestamp is None:
                continue

            # Convert timestamp to microseconds
            milliseconds = bioCam.ClockCyclesToMilliseconds(latest_timestamp)
            microseconds = milliseconds * 1000

            # Check current stimulation state
            stim_type = get_stim_state()

            # Decision logic for sensory stimulation
            if flag == "sensory" and stim_type in ["none", "sensory"]:
                print("Starting or continuing sensory stimulation")
                stim_params, electrode_indices = encode_pong_state(ball_position, paddle_position)
                protocol = prepare_sensory_protocol(protocols, stim_params, electrode_indices)  
                stop_and_reset_protocol()          # stop the previous protocol and reset the stimulator
                set_stim_state("sensory")          # Update stim_state to sensory
                load_and_start_protocol(protocol,"sensory")  # Load and start sensory protocol

            elif flag == "sensory" and stim_type in ["reward", "punishment"]:
                print("Sensory stimulation skipped due to active feedback protocol")

            # Decision logic for reward stimulation
            elif flag == "reward" and stim_type in ["none", "sensory"]:
                print("Starting reward stimulation")
                stop_and_reset_protocol()                          # Stop any current protocol (sensory or previous feedback)
                protocol = protocols[37]                           # Protocol number 38
                set_stim_state("reward")                           # Set stim_state to reward
                load_and_start_theta_burst(protocol, "reward")     # Load reward protocol
                set_stim_state("none")                             # Set stim_state to none
                

            # Decision logic for punishment stimulation
            elif flag == "punishment" and stim_type in ["none", "sensory"]:
                print("Starting punishment stimulation")
                stop_and_reset_protocol()  # Stop any current protocol (sensory or previous feedback)               
                set_stim_state("punishment")  # Set stim_state to punishment
                pulse = pulses[2]
                apply_punishment_stimulation(stimulator, positive_endpoints_full,negative_endpoints_full, pulse)
                set_stim_state("none")                             # Set stim_state to none

            # No sleep in the loop during active stimulations
            if stim_type in ["none", "sensory"]:  # No stimulation ongoing or sensory, can wait
                time.sleep(0.01)  # Allow thread to yield control, adjust if needed for responsiveness


        except Exception as e:
            print(f"Error in stimulation_controller: {e}")

        

# --- Pong Game Logic Thread ---
def pong_game_logic(mea_to_pong_queue, pong_to_viz_queue, pong_to_stim_queue, game_state):
    """
    Main game loop for Pong. Updates ball and paddle positions,
    calculates stimulation parameters, and sends the game state to visualization and stimulation threads.

    Parameters:
    - mea_to_pong_queue: Queue for receiving movement decisions from the MEA processor.
    - pong_to_viz_queue: Queue for sending game state updates to the visualization thread.
    - pong_to_stim_queue: Queue for sending stimulation parameters to the stimulation thread.
    - game_state: Dictionary for storing the state of the game.
    """
    global FPS
    paddle_speed = 0.02  # Adjust speed as needed

    while not stop_event.is_set():
        loop_start = time.time() - program_start
        print(f"Pong thread iteration started at {loop_start:.3f} seconds")

        # Check if there are paddle movement commands in the queue
        try:
            print(f"Queue size before get: {mea_to_pong_queue.qsize()}")
            movement_decision, latest_timestamp = mea_to_pong_queue.get_nowait()
            print(f"Received movement decision: {movement_decision} for Timestamp={latest_timestamp}")
        except queue.Empty:
            movement_decision = None
            latest_timestamp = None

        # Ensure paddle_position is a mutable list
        game_state["paddle_position"] = list(game_state["paddle_position"])

        # Update paddle position based on movement decision
        if movement_decision == "left":
            game_state["paddle_position"][0] -= paddle_speed
            game_state["paddle_position"][0] = max(0.16, game_state["paddle_position"][0])  # Stay in bounds
        elif movement_decision == "right":
            game_state["paddle_position"][0] += paddle_speed
            game_state["paddle_position"][0] = min(1-0.16, game_state["paddle_position"][0])  # Stay in bounds

        # Update the ball's position and check for collisions
        ball_position, ball_velocity, flag = update_ball(
            game_state["ball_position"],
            game_state["ball_velocity"],
            game_state["paddle_position"]
        )

        game_state["ball_position"] = ball_position
        game_state["ball_velocity"] = ball_velocity
        game_state["flag"] = flag

        # Check the current stim_state
        stim_type = get_stim_state()  # Get the current stimulation type from the stim_controller

        if flag == 1:  # **Sensory**
            if stim_type == "none" or stim_type == "sensory":
                # Sensory stimulation can be applied or resumed if no other stim is playing
                print("Sensory stimulation. Updating game state.")
                pong_to_stim_queue.put({
                    "timestamp": latest_timestamp,
                    "ball_position": ball_position,
                    "paddle_position": game_state["paddle_position"],
                    "type": "sensory"
                })

                # Update the game state based on update_ball data
                game_state["ball_position"] = ball_position
                game_state["ball_velocity"] = ball_velocity
                game_state["paddle_position"] = game_state["paddle_position"]

        elif flag == 3:  # **Reward**
            if stim_type == "none" or stim_type == "sensory":
                # If sensory stimulation is active or no stim is active, apply reward stimulation
                print("Reward signal triggered! Sending reward stimulation.")
                pong_to_stim_queue.put({
                    "timestamp": latest_timestamp,
                    "ball_position": ball_position,
                    "paddle_position": game_state["paddle_position"],
                    "type": "reward"
                })        # Send reward signal to stimulator thread

                # Update the game state
                game_state["ball_position"] = ball_position
                game_state["ball_velocity"] = ball_velocity
                game_state["paddle_position"] = game_state["paddle_position"]

        elif flag == 2:  # **Punishment**
            if stim_type == "none" or stim_type == "sensory":

                # Reset the game state (ball and paddle in center)
                game_state["ball_position"] = [0.5, 0.5]  # Reset ball position to center
                game_state["paddle_position"] = [0.5, 0.16]  # Reset paddle position to center
                # --------------- set ball velocity random

                # If sensory stimulation is active or no stim is active, apply punishment
                print("Punishment signal triggered! Sending punishment stimulation.")
                pong_to_stim_queue.put({
                    "timestamp": latest_timestamp,
                    "ball_position": ball_position,
                    "paddle_position": game_state["paddle_position"],
                    "type": "punishment"
                })   

               

        # Send the updated game state to the visualization queue
        if pong_to_viz_queue.qsize() >= pong_to_viz_queue.maxsize:
            pong_to_viz_queue.get()  # Remove the oldest game state
        pong_to_viz_queue.put({
            "timestamp": time.time(),
            "ball_position": ball_position,
            "paddle_position": game_state["paddle_position"],
        })

        print(f"Game updated. Paddle: {game_state['paddle_position']}, Ball: {game_state['ball_position']}")

        # Wait for the next frame
        time.sleep(1 / FPS)
        loop_end = time.time() - program_start
        print(f"Pong thread iteration ended at {loop_end:.3f} seconds, duration: {loop_end - loop_start:.3f} seconds")





        
# --- MEA Data Processing Thread ---

def mea_data_processor(data_queue):
    """
    Processes MEA data: filters motor electrodes, detects spikes, and sends results to Pong and Visualization.
    """

    std_threshold = 2;
    while not stop_event.is_set():
        try:
            # Fetch the latest data chunk from the queue
            raw_data, timestamp = data_queue.get(timeout=0.1)

            if raw_data is None or timestamp is None:
                # No new data, wait briefly and retry
                time.sleep(0.01)
                continue
            callback_start = time.time()
            # Process the data (reshape and filter)
            filtered_data, _ = get_mea_chunk_data(raw_data, timestamp)

            if filtered_data is None:
                continue  # Skip if filtering failed

            # Perform spike detection
            callback_end = time.time()



        except queue.Empty:
            # No new data in the queue
            print("MEA Processor: No data available.")
        except Exception as e:
            print(f"MEA Processing Error: {e}")

def save_streamed_data_preallocated(h5_filename, data_queue, prealloc_done, initial_chunks=2406, batch_size=10):
    """
    Saves streamed MEA data to an HDF5 file using both buffering and preallocation.

    - Each data chunk is a 1D array of size 8089600 (4096 * 1975).
    - Preallocates space for initial_chunks and expands if needed.
    - Buffers a number of chunks (batch_size) before writing to improve I/O performance.
    """
   
    from queue import Empty

    global stop_event

    samples_per_chunk = 4096 * 1975  # Total samples per data chunk
    total_chunks_allocated = initial_chunks  # How many chunks we preallocate
    total_chunks_written = 0  # Tracks how many chunks have been written so far

    # --- Create the HDF5 file and preallocate the datasets ---
    with h5py.File(h5_filename, "w") as h5file:
        # Preallocate 1D dataset for electrode data
        dset_data = h5file.create_dataset(
            "electrode_data",
            shape=(total_chunks_allocated * samples_per_chunk,),
            maxshape=(None,),  # Allow unlimited growth
            dtype='uint16'
        )

        # Preallocate dataset for timestamps (1 per chunk)
        dset_timestamps = h5file.create_dataset(
            "timestamps",
            shape=(total_chunks_allocated,),
            maxshape=(None,),
            dtype='int64'
        )

        print(f"Preallocated space for {total_chunks_allocated} chunks.")
        prealloc_done.set()  # Signal to main thread that preallocation is finished


        # --- Temporary buffers to store incoming data before writing in batches ---
        buffer_data = []
        buffer_timestamps = []

        # --- Main loop for asynchronous saving ---
        while not stop_event.is_set():
            try:
                # Wait for new data and timestamp from the queue
                data, timestamp = data_queue.get(timeout=1)

                # Sanity check on chunk shape
                if data.shape[0] != samples_per_chunk:
                    print(f"Unexpected data shape: {data.shape[0]}, expected {samples_per_chunk}")
                    continue

                # Store data and timestamp in the buffer
                buffer_data.append(data)
                buffer_timestamps.append(timestamp)

                # --- Write to HDF5 in batches ---
                if len(buffer_data) >= batch_size:
                    num_new_chunks = len(buffer_data)
                    total_new_samples = num_new_chunks * samples_per_chunk

                    # --- Resize dataset if not enough space left ---
                    if total_chunks_written + num_new_chunks > total_chunks_allocated:
                        new_alloc = total_chunks_allocated * 2
                        dset_data.resize((new_alloc * samples_per_chunk,))
                        dset_timestamps.resize((new_alloc,))
                        total_chunks_allocated = new_alloc
                        print(f"Resized datasets to hold {total_chunks_allocated} chunks.")

                    # --- Write buffer to dataset ---
                    start_idx = total_chunks_written * samples_per_chunk
                    end_idx = start_idx + total_new_samples

                    dset_data[start_idx:end_idx] = np.concatenate(buffer_data)
                    dset_timestamps[total_chunks_written:total_chunks_written + num_new_chunks] = buffer_timestamps

                    # Update count and clear buffer
                    total_chunks_written += num_new_chunks
                    buffer_data.clear()
                    buffer_timestamps.clear()

            except Empty:
                continue  # No data available right now
            except Exception as e:
                print(f"Error while saving data: {e}")

        print("Data saving stopped.")        
# --- Initialize and Start Threads ---


def main():
    """
    Simplified version with:
    - 2 stimulation protocols (only endpoints differ)
    - Alternates every second
    - Continuous data recording in parallel
    """
    global bioCam, stop_event, protocol_manager, meaPlatePilot
    
    # Initialize BioCam
    stop_event = Event()
    prealloc_done = Event()
    
    initialize_biocam()
    # Access the MeaPlate property (IMeaPlatePilot) of the BioCam instance
    meaPlatePilot = bioCam.MeaPlate
    set_chamber_temperature(37.0)

    # Setup data recording
    data_queue = Queue(maxsize=10)
    timestamp_queue = Queue(maxsize=10)
    initialize_streaming(data_queue, timestamp_queue)

    # Simplified electrode definitions
    positive_electrodes = [(17, 17), (17, 21), (17, 41), (17, 45), (21, 17), (21, 21), (21, 41), (21, 45), (25, 17), (25, 21), (25, 41), (25, 45), (33, 9), (33, 13), (33, 49), (33, 53), (37, 13), (37, 17), (37, 45), (37, 49), (41, 17), (41, 21), (41, 41), (41, 45), (45, 21), (45, 25), (45, 29), (45, 33), (45, 37), (45, 41)]
    negative_electrodes = [(17, 18), (17, 22), (17, 42), (17, 46), (21, 18), (21, 22), (21, 42), (21, 46), (25, 18), (25, 22), (25, 42), (25, 46), (33, 10), (33, 14), (33, 50), (33, 54), (37, 14), (37, 18), (37, 46), (37, 50), (41, 18), (41, 22), (41, 42), (41, 46), (45, 22), (45, 26), (45, 30), (45, 34), (45, 38), (45, 42)]

    alt_positive_electrodes = [(13, 17), (13, 21), (13, 41), (13, 45), (17, 17), (17, 21), (17, 41), (17, 45), (21, 17), (21, 21), (21, 41), (21, 45), (33, 21), (33, 25), (33, 29), (33, 33), (33, 37), (33, 41), (37, 17), (37, 21), (37, 41), (37, 45), (41, 13), (41, 17), (41, 45), (41, 49), (45, 9), (45, 13), (45, 49), (45, 53)]
    alt_negative_electrodes = [(13, 18), (13, 22), (13, 42), (13, 46), (17, 18), (17, 22), (17, 42), (17, 46), (21, 18), (21, 22), (21, 42), (21, 46), (33, 22), (33, 26), (33, 30), (33, 34), (33, 38), (33, 42), (37, 18), (37, 22), (37, 42), (37, 46), (41, 14), (41, 18), (41, 46), (41, 50), (45, 10), (45, 14), (45, 50), (45, 54)]

    # Create stimulation protocols
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

    # Initialize stimulator
    stimulator = bioCam.Stimulator
    stimulator.Initialize()
    stimulator.Start()
    protocol_manager = stimulator.Protocol

    # Create two identical protocols (only endpoints differ)
    protocols = []
    for i in range(2):
        stim_props = create_stim_properties()
        pulse = create_rectangular_pulse(
            name=f"BasicPulse_{i}",
            stim_properties=stim_props,
            amp1=5.0,
            width1=100,
            inter_width=25,
            amp2=-5.0,
            width2=100
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

    # Start data processing thread
    def data_processing_thread():
        """Thread to process incoming data chunks"""
        while not stop_event.is_set():
            try:
                raw_data, timestamp = data_queue.get(timeout=0.1)
                if raw_data is not None:
                    # Simple processing - just show we got data
                    print(f"Received data chunk at {timestamp}, size: {len(raw_data)} bytes")
                    # In real use, you'd do spike detection/processing here
            except queue.Empty:
                continue

    #processing_thread = Thread(target=data_processing_thread, daemon=True)
    #processing_thread.start()
    h5_filename = "train_session.h5"
    saving_thread = Thread(target=save_streamed_data_preallocated, args=(h5_filename, data_queue,prealloc_done ), daemon=True)
    #saving_thread = Thread(target=save_streamed_data, args=(h5_filename, data_queue), daemon=True)
    saving_thread.start()

    print("Waiting for saving thread to be ready...")
    prealloc_done.wait()  # Wait here
    print("OK, ready! Let's start stimulation.")




    try:
        print("Starting alternating stimulation with parallel recording...")
        while not stop_event.is_set():
            for cycle in range(8):
                for trial in range(4):
                    stims=[1,1,1,1,1,2,2,2,2,2]
                    random.shuffle(stims)
                    count=0
                    for stim in stims:
                        count+=1
                        print(f"Cycle {cycle}/ Trial{trial+1} / Stim {count}")
                        if stim==1:
                            stimulator.Send(pulse,endpoints_protocol1[0],endpoints_protocol1[1])
                        else:
                            stimulator.Send(pulse,endpoints_protocol2[0],endpoints_protocol2[1])
                        
                        time.sleep(1)
                    now = datetime.now()
                    print(f"Start 20 second rest trial{trial+1} in cycle{cycle+1}   @ {now.strftime("%H:%M:%S")}")
                    time.sleep(20)
                now = datetime.now()
                print(f"Start 10 min rest for cycle{cycle+1}   @ {now.strftime("%H:%M:%S")}")
                time.sleep(10*60)
            stop_event.set()

                    


    except KeyboardInterrupt:
        print("Experiment stopped by user")
    finally:
        stop_event.set()
        saving_thread.join()
        
        terminate_acquisition()
        print("Experiment terminated")

def create_stim_properties():
    """Create basic stimulation properties"""
    constructor = StimPropertiesType.GetConstructor([
        Int32, Double, Int32, Int32
    ])
    return constructor.Invoke([
        Int32(time_resolution_value),  # time resolution
        Double(amplitude_resolution_value),  # amplitude resolution
        Int32(-25),  # min amplitude
        Int32(25)  # max amplitude
    ])

def create_rectangular_pulse(name, stim_properties, amp1, width1, inter_width, amp2, width2):
    """Create a basic rectangular pulse"""
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
    """Create a basic stimulation protocol"""
    pulse_rate = Double(frequency)
    duration_us = Int32(int(duration * 1e6))
    
    constructor = StimTrainProtocolType.GetConstructor([
        String, rectangularStimPulseType, StimPropertiesType, Int32, Double
    ])
    
    protocol = constructor.Invoke([
        String(name),
        pulse,
        stim_properties,
        Int32(1),  # single pulse
        pulse_rate
    ])

    
    return protocol
if __name__ == "__main__":
    main()

