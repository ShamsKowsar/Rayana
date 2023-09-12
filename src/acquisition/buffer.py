from __future__ import annotations
import numpy as np
from utils import get_lock
from filelock import BaseFileLock
from multiprocessing.shared_memory import SharedMemory
from shared_memory_dict import SharedMemoryDict


class BufferAccess:
    _buffer:np.ndarray
    _write_cursor:SharedMemoryDict|dict
    _read_timestamp:int = 0

    def __init__(self,buffer:np.ndarray,write_cursor:SharedMemoryDict|dict) -> None:
        self._buffer = buffer
        self._write_cursor = write_cursor

    def write(self,val):
        count = len(val)
        buffer_length = len(self._buffer)
        if count > buffer_length:
            raise OverflowError
        max_timestamp = self._write_cursor['max_timestamp']
        min_timestamp = self._write_cursor['min_timestamp']
        max_timestamp_index = self._write_cursor['max_timestamp_index']
        if max_timestamp_index + count < buffer_length:
            self._buffer[max_timestamp_index+1:max_timestamp_index+1+count] = val
        else:
            self._buffer[max_timestamp_index+1:] = val[:buffer_length-(max_timestamp_index+1)]
            self._buffer[:count-(buffer_length-(max_timestamp_index+1))] = val[buffer_length-(max_timestamp_index+1):]
        self._write_cursor['max_timestamp'] += count
        self._write_cursor['max_timestamp_index'] = (max_timestamp_index + count) % buffer_length
        if(max_timestamp>=buffer_length):
            self._write_cursor['min_timestamp'] = max_timestamp - buffer_length

    def read(self,n:int,dest:np.ndarray,offset:int = 0):
        buffer_length = len(self._buffer)
        max_timestamp = self._write_cursor['max_timestamp']
        min_timestamp = self._write_cursor['min_timestamp']
        max_timestamp_index = self._write_cursor['max_timestamp_index']
        if min_timestamp <= self._read_timestamp:
            if self._read_timestamp + n <= max_timestamp:
                read_timestamp_index = (max_timestamp_index - (max_timestamp-self._read_timestamp)) % buffer_length
                if read_timestamp_index + n < buffer_length:
                    dest[offset:offset+n] = self._buffer[read_timestamp_index:read_timestamp_index+n]
                else:
                    dest[offset:offset+(buffer_length-read_timestamp_index)] = self._buffer[read_timestamp_index:]
                    dest[offset+(buffer_length-read_timestamp_index):offset+n] = self._buffer[:(read_timestamp_index+n)%buffer_length]
            else:
                return False
        else:
            raise OverflowError
        self._read_timestamp += n
        return True

    
class WritableSharedBuffer(BufferAccess):
    _shared_mem:SharedMemory
    _lock:BaseFileLock
    _write_cursor:SharedMemoryDict

    def __init__(self,streamname:str,shape:tuple,sample_dtype):
        self._shared_mem = SharedMemory(streamname,create=True,size=self._get_array_size(shape,sample_dtype))
        buffer = np.ndarray(shape,sample_dtype,buffer=self._shared_mem.buf)
        self._lock = get_lock(streamname)
        super().__init__(buffer, self._init_write_cursor(streamname,shape,sample_dtype))

    def _init_write_cursor(self,streamname:str,shape:tuple,sample_dtype):
        write_cursor = SharedMemoryDict(streamname+'_wcursor',1024)
        with self._lock:
            write_cursor['max_timestamp'] = -1
            write_cursor['min_timestamp'] = -1
            write_cursor['max_timestamp_index'] = -1
            write_cursor['shape'] = shape
            write_cursor['dtype'] = sample_dtype
            write_cursor['finished'] = False
        return write_cursor

    @staticmethod
    def _get_array_size(shape:tuple,sample_dtype):
        size:int = 1
        for dim in shape:
            size *= dim
        size *= np.dtype(sample_dtype).itemsize
        return size
    
    def read(self, n: int, dest: np.ndarray, offset: int = 0):
        with self._lock:
            return super().read(n, dest, offset)
        
    def write(self, val):
        with self._lock:
            return super().write(val)
        
    def set_finished(self):
        with self._lock:
            self._write_cursor['finished'] = True
    
    def __del__(self):
        self._shared_mem.close()
        self._shared_mem.unlink()
        self._write_cursor.shm.close()
        self._write_cursor.shm.unlink()
        
class ReadonlySharedBuffer(BufferAccess):
    _shared_mem:SharedMemory
    _lock:BaseFileLock

    def __init__(self, streamname:str) -> None:
        self._shared_mem = SharedMemory(streamname)
        write_cursor = SharedMemoryDict(streamname+'_wcursor',1024)
        buffer = np.ndarray(write_cursor['shape'],write_cursor['dtype'],buffer=self._shared_mem.buf)
        self._lock = self._lock = get_lock(streamname)

        super().__init__(buffer, write_cursor)

    def write(self, val):
        raise NotImplementedError()
    
    def read(self, n: int, dest: np.ndarray, offset: int = 0):
        with self._lock:
            return super().read(n, dest, offset)
        
    @property
    def is_finished(self)->bool:
        with self._lock:
            return self._write_cursor['finished']
        
    def __del__(self):
        self._shared_mem.close()
        self._write_cursor.shm.close() # type: ignore
        
    