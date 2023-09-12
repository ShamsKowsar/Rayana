import os
from time import perf_counter
from io import BufferedReader,BufferedWriter
from pathlib import Path
from subprocess import Popen
import numpy as np
from abc import ABCMeta, abstractmethod

class Timer:
    _start_time:float
    _running:bool = False

    def restart(self):
       self._start_time = perf_counter()
       self._running = True
    
    def elapsed(self):
       return perf_counter() - self._start_time
    
    @property
    def running(self):
        return self._running
       
    def stop(self):
        self._running = False

class MEA_Stream(metaclass=ABCMeta):
    _chunk_size:int
    def __init__(self,chunk_size):
        self._chunk_size = chunk_size

    @property
    @abstractmethod
    def recording_parameters(self):
        pass

    @abstractmethod
    def read(self):
        pass

    @property
    @abstractmethod
    def is_finished(self):
        pass


class MeabenchStream(MEA_Stream):
    SAMPLE_SIZE = 2
    CHANNELS = 64
    MEABENCH_PATH = '/usr/local/share/meabench/bin'

    _chunk_size:int
    _file_path:str
    _pipe_stream:BufferedWriter
    _recording_file_stream:BufferedReader

    _flush_wait_count:int = 0
    FLUSH_WAIT_THRESHOLD:int = 3

    _ending_wait_timer:Timer
    ENDING_WAIT_THRESHOLD = 0.1 #seconds

    def __init__(self,chunk_size,mode):
        self._ending_wait_timer = Timer()

        self._recording_filepath = self._generate_unique_recording_name()
        stream_type = 'raw' if mode == 'realtime' else 'reraw'
        self._pipe_stream = open(self._recording_filepath,mode='wb')
        Popen([Path(self.MEABENCH_PATH).joinpath('stream2pipe'),'raw',stream_type],stdout=self._pipe_stream)
        self._recording_file_stream = open(self._recording_filepath,'rb')
        super().__init__(chunk_size)

    @property
    def recording_parameters(self):
        return {'frequency':25000,'gain':0.333496,'digital_zero':2048,'channels':self.CHANNELS,'dtype':np.short}

    @staticmethod
    def _generate_unique_recording_name():
        index = 0
        prefix = Path().cwd().joinpath('recordings')
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        while(True):
            record_path = prefix.joinpath(f'recording_{index}')
            if not os.path.isfile(record_path):
                return record_path
            index += 1

    @property
    def _remaining_samples(self):
        current_cursor = self._recording_file_stream.tell()
        self._recording_file_stream.seek(0,os.SEEK_END)
        current_size = self._recording_file_stream.tell()
        self._recording_file_stream.seek(current_cursor)
        return (current_size - current_cursor) // (self.SAMPLE_SIZE*self.CHANNELS)

    def wait_for_start(self):
        while self._remaining_samples == 0:
            pass

    def read(self):
        remaining_samples = self._remaining_samples
        if remaining_samples >= self._chunk_size:
            self._flush_wait_count = 0
            self._ending_wait_timer.stop()
            return np.fromfile(self._recording_file_stream,np.short,count=self._chunk_size*self.CHANNELS)
        else:
            self._flush_wait_count += 1
            if self._flush_wait_count >= self.FLUSH_WAIT_THRESHOLD and remaining_samples != 0:
                self._ending_wait_timer.stop()
                return np.fromfile(self._recording_file_stream,np.short,count=remaining_samples*self.CHANNELS)
            if remaining_samples == 0 and self._ending_wait_timer.running == False:
                self._ending_wait_timer.restart()
            return None
    
    def is_finished(self):
        if self._ending_wait_timer.running:
            return self._ending_wait_timer.elapsed() > self.ENDING_WAIT_THRESHOLD
        return False
    
    def __del__(self):
        self._recording_file_stream.close()
        self._pipe_stream.close()
