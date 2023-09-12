import asyncio
import numpy as np
from io import BytesIO
from multiprocessing.shared_memory import SharedMemory
from shared_memory_dict import SharedMemoryDict
from buffer import WritableSharedBuffer
import pickle

SERVER_PORT = 8888
SERVER_IP = '127.0.0.1' #localhost

def create_byte_stream(message:bytes):
    temp_buffer = BytesIO()
    temp_buffer.write(message)
    temp_buffer.seek(0)
    return temp_buffer

async def get_recording_params(reader:asyncio.StreamReader,end_postfix:bytes) -> SharedMemoryDict:
    rec_params_pickle:bytes = (await reader.readuntil(end_postfix)).rstrip(end_postfix)
    rec_params:dict = pickle.loads(rec_params_pickle)
    shared_rec_params = SharedMemoryDict('raw_params',1024)
    for key,value in rec_params.items():
        shared_rec_params[key] = value
    return shared_rec_params

async def handle_connection():
    try:
        reader, _ = await asyncio.open_connection(SERVER_IP, SERVER_PORT)
    except ConnectionRefusedError:
        print('unable to connect')
        raise SystemExit
    obj_end_postfix = 'end'.encode()
    stream_end_postfix = 'stream_end'.encode()
    shared_recording_params:SharedMemoryDict = await get_recording_params(reader,obj_end_postfix)
    buffer = WritableSharedBuffer('raw',(120*25000,64),np.short)
    print('connection established ...') 

    while(True):
        recieved = await reader.readuntil(obj_end_postfix)
        if recieved == stream_end_postfix:
            buffer.set_finished()
            print('streaming finished')
            break
        np_samples = np.load(create_byte_stream(recieved))
        # print(np_samples)
        buffer.write(np_samples)
    shared_recording_params.shm.close()
    shared_recording_params.shm.unlink()

print(f'connecting to server at {SERVER_IP}, port {SERVER_PORT} ...')

asyncio.run(handle_connection())