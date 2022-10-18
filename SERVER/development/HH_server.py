#!/bin/python
#
# inspired by
# https://github.com/furas/python-examples/blob/master/socket/basic/version-4/server.py
#

import json
import socket
import requests
import argparse
import traceback
import threading
import multiprocessing
from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import librosa

import data_processors as dp
from networks import LinearNetwork, VAENetwork


parser = argparse.ArgumentParser()
parser.add_argument("port", type=int)

args = parser.parse_args()

HOST = "0.0.0.0"
PORT = 8080
if args.port:
    PORT = args.port

STIM_ADDR = "192.168.22.107"

SR = 1024
FRAME_LENGTH = 512
HOP_LENGTH = 128
FR = librosa.samples_to_frames(SR, hop_length=HOP_LENGTH)

INPUT_MAX = 4096
#INPUT_MAX = 1024

BUFFERSIZE = 1024
MAX_PLOT_SAMPLES = 4000
MAX_PLOT_FRAMES = librosa.samples_to_frames(MAX_PLOT_SAMPLES, hop_length=HOP_LENGTH)


DATA_KEYS = ["data0", "data1", "data2", "data3"]
DATA_NAME_MAP = {
    "data0": "EEG",
    "data1": "EOG",
    "data2": "heart",
    "data3": "GSR",
}

AX_LIMS = {
    "EEG": (-1, 1),
    #"EEG": (0, 200),
    "EOG": (-1, 1),
    "GSR": (0, INPUT_MAX),
    "heart": (0, INPUT_MAX),
}

t_sr = librosa.samples_to_time(np.arange(MAX_PLOT_SAMPLES) - MAX_PLOT_SAMPLES, sr=SR)
t_f = librosa.frames_to_time(
    np.arange(MAX_PLOT_FRAMES) - MAX_PLOT_FRAMES, sr=SR, hop_length=HOP_LENGTH
)

processors = {
    "EEG": dp.ProcessorList(
        dp.RecordProcessor(sr=SR, fn=f"eeg_raw_{SR}hz.txt"),
        dp.NormaliseProcessor(sr=SR, inrange=(0, INPUT_MAX), max_hist=MAX_PLOT_SAMPLES*16),
        #dp.RecordProcessor(sr=SR, fn=f"eeg_norm_{SR}hz.txt"),
        dp.FilterProcessor(sr=SR, Wn=(25, 45), btype="bandpass"),
        #dp.RecordProcessor(sr=SR, fn=f"eeg_filtered_{SR}hz.txt"),
        dp.RMSProcessor(sr=SR, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH),
        #dp.RecordProcessor(sr=SR, fn=f"eeg_rms_{SR}hz.txt"),
    ),
    "EOG": dp.ProcessorList(
        dp.NormaliseProcessor(sr=SR, inrange=(0, INPUT_MAX), max_hist=MAX_PLOT_SAMPLES),
    ),
    "GSR": dp.ProcessorList(),
    "heart": dp.ProcessorList(),
}

feature_extractors = {
    "EEG": dp.FeatureExtractorCollection(
        dp.MeanFeature(sr=FR, units="frames", period=FR),
        dp.DeltaFeature(sr=FR, units="frames", period=FR),
    ),
    "EOG": dp.FeatureExtractorCollection(
        dp.PeakActivityFeature(sr=SR, units="samples", period=SR),
    ),
    "GSR": dp.FeatureExtractorCollection(
        dp.MeanFeature(sr=SR, units="samples", period=SR),
        dp.DeltaFeature(sr=SR, units="samples", period=SR),
    ),
    "heart": dp.FeatureExtractorCollection(
        dp.StdDevFeature(sr=SR, units="samples", period=SR),
    ),
}

last_feature_time = 0
last_output_time = 0

data_units = {
    "data0": "frames",
    "data1": "samples",
    "data2": "samples",
    "data3": "samples"
}

fig, axes = plt.subplots(4, sharex=True)

data = dict()
threads = []
connections = dict()

FEATURE_VECTOR_MAP = [
    "EEG:mean",
    "EEG:delta",
    "EOG:all_rate",
    "EOG:p_rate",
    "EOG:n_rate",
    "GSR:mean",
    "GSR:delta",
    "heart:std",
]

OUTPUT_VECTOR = ["ampl", "freq", "durn", "idly"]

OUTPUT_VECTOR_RANGES = {
    "ampl": [0, 12],
    "freq": [1, 100],
    "durn": [0, 2000],
    "idly": [0, 255],
}

embedder_params = {
    "feature_size": 8, 
    "hidden_size": [16, 8],
    "latent_size": 2,
}

decision_params = {
    "sizes": [2, 8, 8, 4],
    "batchnorm": True,
}


embedderNetwork = VAENetwork(embedder_params)
interpretterNetwork = LinearNetwork(decision_params)


def get_request(url):
    res = requests.get(url)
    if not res.ok:
        print(url, "returned status", res.status_code)


def makeFeatureVector(features):
    x = [features.get(f, 0) for f in FEATURE_VECTOR_MAP]
    x = np.array(x)

    return x

def sendOutput(host, y):
    global last_output_time

    now = time()
    if now - last_output_time < 4.0:
        return

    last_output_time = now

    if host not in connections:
        print(host, "not in connections")
        print(connections)
        return

    chan = connections[host] + 1
    url = f'http://{STIM_ADDR}/enab/{chan}/1'
    print(url)
    t = threading.Thread(target=send_output, args=(url,))
    t.start()

    for i, name in enumerate(OUTPUT_VECTOR):
        val = y[i]
        r = OUTPUT_VECTOR_RANGES[name]
        val = val * (r[1] - r[0]) + r[0]

        url = f'http://{STIM_ADDR}/{name}/{chan}/{int(val)}'
        print(url)
        t = threading.Thread(target=send_output, args=(url,))
        t.start()

        #send_output(url)

    url = f'http://{STIM_ADDR}/stim/{chan}/{4}'
    print(url)
    t = threading.Thread(target=send_output, args=(url,))
    t.start()
    #send_output(url)


def collect(q):
    if q.empty():
        return

    while not q.empty():
        (host, port), values_dict = q.get()
        if host not in connections:
            connections[host] = len(connections)

        if host not in data:
            data[host] = dict()
            for dk in DATA_KEYS:
                name = DATA_NAME_MAP[dk]
                if data_units[dk] == "samples":
                    data[host][name] = np.zeros(MAX_PLOT_SAMPLES)
                else:
                    data[host][name] = np.zeros(MAX_PLOT_FRAMES)

        for dk in DATA_KEYS:
            if dk not in values_dict:
                continue
            values = values_dict[dk]
            name = DATA_NAME_MAP[dk]
            if data_units[dk] == "samples":
                max_hist = MAX_PLOT_SAMPLES
            else:
                max_hist = MAX_PLOT_FRAMES
            data[host][name] = np.concatenate([data[host][name], values])[-max_hist:]

    getFeatures()


def getFeatures():
    global last_feature_time

    now = time()
    if now - last_feature_time < 1.0:
        return
    last_feature_time = now

    all_features = dict()

    for host in data.keys():
        features = dict()
        for name in DATA_NAME_MAP.values():
            buffer = data[host][name]
            result = feature_extractors[name](buffer)
            for k, v in result.items():
                f_name = name + ":" + k
                features[f_name] = v

        all_features[host] = features

    for host in all_features.keys():
        x = makeFeatureVector(all_features[host])
        #print(x)
        z = embedderNetwork(x)
        #print(z, z.shape)

        y = interpretterNetwork(z)[0]
        print(y, y.shape)
        sendOutput(host, y)

    print("-----------")


def send_output(url):
    headers = {
        "Connection": "close",
    }
    try:
        res = requests.get(url, headers=headers, timeout=0.5)
    except requests.exceptions.ConnectTimeout:
        return

    if not res.ok:
        print(f"request {url} returned {r.status_code}")
    res.close()


def animate(i, q):
    global axes, data

    collect(q)

    for ax in axes:
        ax.clear()

    for i, (dk, ax) in enumerate(zip(DATA_KEYS, axes)):
        name = DATA_NAME_MAP[dk]
        for host in data.keys():

            #ax.hline(0, color="black", lw=1, ls="--")
            if data_units[dk] == "samples":
                ax.plot(t_sr, data[host][name], label=host, lw=1)
            else:
                ax.plot(t_f, data[host][name], label=host, lw=1)


        ax.set_ylim(*AX_LIMS[name])
        if len(data) > 0 and i == 0:
            ax.legend(loc="upper right")
        ax.set_title(name)


def plot(q):
    ani = animation.FuncAnimation(fig, animate, fargs=(q,), interval=250)
    plt.show()


def handle_client(q, conn, addr):
    try:
        with conn:
            print()
            print("*" * 20)
            print(f"made connection to {addr}")
            print("*" * 20)
            print()
            #connections[addr[0]] = len(connections)
            data = b""
            while True:
                msg = conn.recv(BUFFERSIZE)
                data = data + msg
                if "#" not in msg.decode("ascii"):
                    continue

                lines = data.decode("ascii").split("#")
                for line in lines[:-1]:
                    values = processBuffer(line)
                    for dk, v in values.items():
                        name = DATA_NAME_MAP[dk]
                        if name not in processors:
                            continue

                        values[dk] = processors[name](np.array(v, dtype=float))

                    q.put((addr, values))
                data = bytes(lines[-1], "ascii")

    except Exception as e:
        print(f"[DEBUG] exception from {addr}")
        traceback.print_exc()
        print(e)

        conn.close()


def processBuffer(data):
    #print(data)
    try:
        values = json.loads(data)
        return values
    except Exception as e:
        return {"data0": [], "data1": [], "data2": [], "data3": []}


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

        print(f"Server listening on {HOST}:{PORT}")

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
            s.bind(("", PORT))
            print("lighthouse started on", PORT)
        except OSError:
            print(PORT, "already bound...")
            s.close()
            return

        while True:
            msg, addr = s.recvfrom(BUFFERSIZE)
            #if msg == b"HH":
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
