#!/bin/python
#
# inspired by
# https://github.com/furas/python-examples/blob/master/socket/basic/version-4/server.py
#

import json
import socket
import threading
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

HOST = "0.0.0.0"
PORT = 8081

# INPUT_MAX = 4096
INPUT_MAX = 1024

BUFFERSIZE = 1024*4
MAX_PLOT = 4000
DATA_KEYS = ["data0", "data1", "data2"]
AX_LIMS = {
    "data0": (0, INPUT_MAX),
    "data1": (0, INPUT_MAX),
    "data2": (0, INPUT_MAX),
}

fig, axes = plt.subplots(3, sharex=True)

data = dict()
threads = []
connections = []

def animate(i, q):
    global axes, data
    while not q.empty():
        (host, port), values_dict = q.get()
        #print(values_dict)

        if host not in data:
            data[host] = dict()
            for dk in DATA_KEYS:
                data[host][dk] = np.zeros(MAX_PLOT)

        for dk in DATA_KEYS:
            if dk not in values_dict:
                continue
            values = values_dict[dk]
            data[host][dk] = np.concatenate([data[host][dk], values])[-MAX_PLOT:]

    for ax in axes:
        ax.clear()

    for i, (dk, ax) in enumerate(zip(DATA_KEYS, axes)):
        for host in data.keys():
            ax.plot(data[host][dk], label=host)
            print(dk, data[host][dk][-8:])
        ax.set_ylim(*AX_LIMS[dk])
        #print('connections', len(connections))
        if len(data) > 0 and i == 0:
            ax.legend(loc='upper right')
        ax.set_title(dk)


def plot(q):
    ani = animation.FuncAnimation(fig, animate, fargs=(q,), interval=200)
    plt.show()


def handle_client(q, conn, addr):
    try:
        with conn:
            print(f'made connection to {addr}\n')
            connections.append(addr[0])
            data = ""
            while True:
                msg = conn.recv(BUFFERSIZE).decode("ascii")
                data = data + msg
                if "#" in msg:
                    lines = data.split('#')
                    values = processBuffer(lines[0])
                    q.put((addr, values))
                    data = lines[-1]

    except Exception as e:
        print(f'[DEBUG] exception from {addr}')
        print(e)

        conn.close()


def processBuffer(data):
    #print("-------")
    #print(data)
    #print("-------")
    try:
        values = json.loads(data)
        return values
    except Exception as e:
        print(f'[DEBUG] exception:')
        print(e)
        return {"data0": [], "data1": [], "data2": []}


def server(q):
    print("starting server thread...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, PORT))
        except OSError:
            print(HOST, PORT, "already bound...")
            s.close()
            return

        s.listen()

        print(f'Server listening on {HOST}:{PORT}')

        try:
            while True:
                conn, addr = s.accept()

                t = threading.Thread(target=handle_client, args=(q, conn, addr))
                t.start()
                threads.append(t)

        except KeyboardInterrupt:
            for t in threads:
                t.join()

            s.close()

def lighthouse():
    print("starting lighthouse")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.bind(('', PORT))
            print("lighthouse started on ", PORT)
        except OSError:
            print(PORT+1, "already bound...")
            s.close()
            return

        while True:
            msg, addr =  s.recvfrom(BUFFERSIZE)
            if msg == b'HH':
                s.sendto(msg, addr)



def main():
    q = multiprocessing.Queue()

    server_process = multiprocessing.Process(None, server, args=(q,))
    server_process.start()
    lighthouse_process = multiprocessing.Process(None, lighthouse)
    lighthouse_process.start()
    plot(q)
    print("finished setup")


if __name__ == "__main__":
    main()
