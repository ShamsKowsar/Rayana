from filelock import FileLock,BaseFileLock
import os
from pathlib import Path

def get_lock(streamname:str)->BaseFileLock:
    lock_name = streamname + '.lock'
    lock_path = Path().home().joinpath('.rayana')
    if not os.path.exists(lock_path):
        os.mkdir(lock_path)
    return FileLock(lock_path.joinpath(lock_name),timeout=1)