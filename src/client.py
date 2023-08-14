import matplotlib.pyplot as plt
import numpy as np
import socket
import pickle
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import random

from Spike_detection_newVersion import *
chunk = 0
HOST = 'localhost'
PORT = 50007
x = np.linspace(0, 6*np.pi, 100)
y = np.sin(x)
spikes=[[] for i in range(60)]
plt.ion()

fig = plt.figure()
ax2 = fig.add_subplot(212)

ax = fig.add_subplot(211)

heatmap = np.random.random(( 6,10 ))
#print(heatmap)
heatmap=[[0 for i in range(10)] for j in range(6)]
label=[[10*j+i+1 for i in range(10)] for j in range(6)]

shw=ax2.imshow( heatmap , cmap = 'GnBu' , interpolation = 'nearest' )
for i in range(6):
    for j in range(10):
        text = ax2.text(j, i, label[i][j],
                    ha="center", va="center", color="k") 
cbar=plt.colorbar(shw,ax=ax2)
# cbar.set_ticks([0,0.5,1])
cbar.set_ticklabels(np.linspace(0,np.max(np.array(heatmap)),5))    
plt.show()

line1, = ax.plot(x, y, 'b-') 
## print('hi')
for i in range(100):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    s.connect((HOST, PORT))
    s.sendall(str(chunk).encode())
    i = 0
    datas = b''
    c = time.time()

    data = s.recv(1024*1024*1024)
    datas += (data)
    while (len(data) >= 1024*1024*1024):
        data = s.recv(1024*1024*1024)
        datas += (data)
    data = json.loads(datas.decode())
    freq=data.get('f')
    data=data.get('m')
#    print(time.time()-c)
    for i in range(60):
        u,v=(spike_detection(data[i],freq))
        spikes[i].extend(v)
        heatmap[int(i/10)][i%10]=len(v)
    line1.set_ydata(data[0])
    line1.set_xdata(np.linspace(chunk,chunk+1,len(data[0])))
    ax.set_xlim(chunk,chunk+1)
    ax.set_ylim(-10,10)
    label=[[random.randrange(10) for i in range(6)] for j in range(10)]
    ax2.imshow( heatmap , cmap = 'GnBu' , interpolation = 'nearest')
    cbar.set_ticklabels(np.linspace(0,np.max(np.array(heatmap)),5))    
    fig.canvas.draw()
    fig.canvas.flush_events()

    chunk+=1