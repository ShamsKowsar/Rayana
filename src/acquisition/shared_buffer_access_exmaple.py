from multiprocessing.shared_memory import SharedMemory
from shared_memory_dict import SharedMemoryDict
from buffer import ReadonlySharedBuffer
import numpy as np

STREAM_NAME:str = 'raw'

def wait_for_init(stream_name:str):
    while True:
        try:
            SharedMemory(stream_name)
            break
        except:
            continue

wait_for_init(STREAM_NAME)
buffer = ReadonlySharedBuffer(STREAM_NAME)
recording_params = SharedMemoryDict(STREAM_NAME+'_params',1024)
dest = np.empty((20,recording_params['channels']),recording_params['dtype'])
while(not buffer.is_finished):
    success = buffer.read(len(dest),dest)
    if success:
        print(dest)

recording_params.shm.close()
