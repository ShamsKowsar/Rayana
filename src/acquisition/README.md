# Acquisition & loading of raw mea stream into shared memory

## Requirements:
- pip install shared-memory-dict
- pip install filelock
- numpy
- preferably python 3.9+ (not sure about exact requirement)

## Usage:
Do the following steps in order:
- run _server.py_
- run _client.py_
- using module _replay_ of meabench, play a recording file (such as the one placed in recordings directory, named _sample_rec.raw_) 

Then the recording will be simultaneously written in shared memory and _recordings_ directory.

To access the shared memory, you can use _shared_buffer_access_example.py_ as a refrence. This script should be run before _server.py_.
