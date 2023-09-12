import os
from io import BytesIO
import numpy as np
import asyncio
import pickle
from mea_stream import MeabenchStream

SERVE_PORT = 8888
SERVE_IP = '127.0.0.1' #localhost


async def handle_client(reader:asyncio.StreamReader,writer:asyncio.StreamWriter):
    print('connection established ...')
    stream = MeabenchStream(20,'replay')
    writer.write(pickle.dumps(stream.recording_parameters))
    writer.write('end'.encode())
    await writer.drain()
    stream.wait_for_start()
    while(not stream.is_finished()):
        samples = stream.read()
        if samples is not None:
            samples_count = len(samples)//64
            samples = np.reshape(samples,(samples_count,64))
            buffer = BytesIO()
            np.save(buffer,samples,allow_pickle=True)
            buffer.seek(0)
            writer.write(buffer.read())
            writer.write('end'.encode())
            await writer.drain()
    writer.write('stream_end'.encode())
    await writer.drain()
    writer.close()
    await writer.wait_closed()
    print('streaming finished')

async def main():
    server = await asyncio.start_server(handle_client, SERVE_IP, SERVE_PORT)
    print(f'starting server at {SERVE_IP}, port {SERVE_PORT} ...')
    await server.serve_forever()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print('quited')
    

