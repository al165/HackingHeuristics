#!/bin/python

import os
import json
import socket
import argparse
import traceback
import threading
import multiprocessing
from queue import Empty
from time import time, sleep
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

import requests
from requests.adapters import Retry, HTTPAdapter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from librosa import samples_to_frames, samples_to_time, frames_to_time

import data_processors as dp
from networks import LinearNetwork, VAENetwork, SACModel

import torch
import torch.nn as nn

from plotters import FullPlotter, MinPlotter

from config import (
    i2c_ADDRESSES,
    STIM_ADDR,
    CHECKPOINT_INTERVAL,
    CHECKPOINT_PATH,
    DATA_KEYS,
    DATA_UNITS,
    DATA_NAME_MAP,
    FEATURE_VECTOR_MAP,
    OUTPUT_VECTOR,
    OUTPUT_VECTOR_RANGES,
)

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8080, required=False)
parser.add_argument(
    "--load",
    action="store_true",
    default=False,
    required=False,
    help="load latest model. Default False.",
)

args = parser.parse_args()

HOST = "0.0.0.0"
PORT = 8080
if args.port:
    PORT = args.port

SR = 512
FRAME_LENGTH = 512
HOP_LENGTH = 128
FR = samples_to_frames(SR, hop_length=HOP_LENGTH)

INPUT_MAX = 4096

BUFFERSIZE = 1024
MAX_PLOT_SAMPLES = 4000
MAX_PLOT_FRAMES = samples_to_frames(MAX_PLOT_SAMPLES, hop_length=HOP_LENGTH)


UPDATE_TIME = 1

t_sr = samples_to_time(np.arange(-MAX_PLOT_SAMPLES, 0), sr=SR)
t_f = frames_to_time(np.arange(-MAX_PLOT_FRAMES, 0), sr=SR, hop_length=HOP_LENGTH)

processors = {
    "EEG": dp.ProcessorList(
        dp.RecordProcessor(sr=SR, fn=f"eeg_raw_{SR}hz.txt"),
        dp.NormaliseProcessor(
            sr=SR, inrange=(0, INPUT_MAX), max_hist=MAX_PLOT_SAMPLES * 16
        ),
        # dp.RecordProcessor(sr=SR, fn=f"eeg_norm_{SR}hz.txt"),
        dp.FilterProcessor(sr=SR, Wn=(25, 45), btype="bandpass"),
        # dp.RecordProcessor(sr=SR, fn=f"eeg_filtered_{SR}hz.txt"),
        dp.RMSProcessor(sr=SR, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH),
        # dp.RecordProcessor(sr=SR, fn=f"eeg_rms_{SR}hz.txt"),
        dp.RescaleProcessor(sr=FR, in_range=(0, 1), out_range=(-1, 1)),
    ),
    "EOG": dp.ProcessorList(
        # dp.NormaliseProcessor(sr=SR, inrange=(0, INPUT_MAX), max_hist=MAX_PLOT_SAMPLES),
        dp.RescaleProcessor(sr=SR, in_range=(0, INPUT_MAX), out_range=(-1, 1))
    ),
    "GSR": dp.ProcessorList(
        dp.RescaleProcessor(sr=SR, in_range=(0, INPUT_MAX), out_range=(-1, 1))
    ),
    "heart": dp.ProcessorList(
        dp.RescaleProcessor(sr=SR, in_range=(0, INPUT_MAX), out_range=(-1, 1))
    ),
}

feature_extractors = {
    "EEG": dp.FeatureExtractorCollection(
        dp.MeanFeature(sr=FR, units="frames", period=UPDATE_TIME * FR),
        dp.DeltaFeature(sr=FR, units="frames", period=UPDATE_TIME * FR),
    ),
    "EOG": dp.FeatureExtractorCollection(
        dp.PeakActivityFeature(sr=SR, units="samples", period=UPDATE_TIME * SR),
    ),
    "GSR": dp.FeatureExtractorCollection(
        dp.MeanFeature(sr=SR, units="samples", period=UPDATE_TIME * SR),
        dp.DeltaFeature(sr=SR, units="samples", period=UPDATE_TIME * SR),
    ),
    "heart": dp.FeatureExtractorCollection(
        dp.StdDevFeature(sr=SR, units="samples", period=UPDATE_TIME * SR),
    ),
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

translator_params = {
    "state_dim": embedder_params["latent_size"] * 2,
    "action_dim": len(OUTPUT_VECTOR),
    "gamma": 0.99,
    "hid_shape": (12, 12),
    "a_lr": 0.0001,
    "c_lr": 0.0001,
    "batch_size": 1,
    "alpha": 0.12,
    "adaptive_alpha": False,
}


@dataclass
class Agent:
    ip: str
    port: int
    mac: str
    id: int
    map: deque = deque(maxlen=8)
    active: bool = False
    highlight: bool = False
    stim_addr: int = 0x1A
    stim_chan: int = 1
    sensor_data: Dict = field(default_factory=dict)
    feature_vectors: deque = deque(maxlen=8)
    output_vectors: deque = deque(maxlen=8)

    def __post_init__(self):
        self.map.append(np.array([0.0, 0.0]))


class Translator(multiprocessing.Process):
    def __init__(
        self,
        sensor_q: multiprocessing.Queue,
        plot_q: multiprocessing.Queue,
        embedder_params: Dict,
        decision_params: Dict,
        translator_params: Dict,
        load_latest: bool = False,
    ):

        super(Translator, self).__init__()
        self.sensor_q = sensor_q
        self.plot_q = plot_q

        self.last_feature_time = 0
        self.last_output_time = dict()

        self.last_checkpoint_time = time()

        self.embedderNetwork = VAENetwork(embedder_params)
        self.translator = SACModel(**translator_params)

        self.targets = deque(maxlen=8)

        self.running = multiprocessing.Event()

        self.agents = dict()  # {host: agent}

        if load_latest:
            self.loadLatest()

        print("Translator init done")

    @staticmethod
    def makeFeatureVector(features) -> np.ndarray:
        x = [features.get(f, 0) for f in FEATURE_VECTOR_MAP]
        x = np.array(x)

        return x

    def getCenter(self) -> np.ndarray:
        """Returns the coordinate of the center of mass of all the points"""
        positions = []
        for host, agent in self.agents.items():
            positions.append(agent.map[-1])

        if len(positions) == 0:
            return np.ndarray([0, 0])

        return np.mean(positions, axis=0)

    def postOutput(self, host: str, y: np.ndarray):
        if host not in self.last_output_time:
            self.last_output_time[host] = 0

        if host not in self.agents:
            print(host, "not in agents")
            return

        now = time()
        if now - self.last_output_time[host] < 4.0:
            return

        self.last_output_time[host] = now

        agent = self.agents[host]

        url = f"http://{STIM_ADDR}/stim"
        headers = {"Content-Type": "application/json"}
        data = {
            "channel": agent.stim_chan,
            "addr": agent.stim_addr,
        }

        host_url = f"http://{host}/update"
        update_data = {
            "hightlight": agent.highlight,
        }

        for i, name in enumerate(OUTPUT_VECTOR):
            if name in ("temp1", "temp2", "highlight"):
                update_data[name.lower()] = val
            else:
                val = np.clip(y[i], -1, 1)
                r = OUTPUT_VECTOR_RANGES[name]
                out = int((val / 2 + 0.5) * (r[1] - r[0]) + r[0])
                data[name.lower()] = out

        t1 = threading.Thread(
            target=post_request,
            args=(url, headers, data, 0.5),
        )
        t1.start()

        t2 = threading.Thread(
            target=post_request,
            args=(host_url, headers, update_data, 2.0),
        )
        t2.start()

    def create_agent(self, mac: str, host: str, port: int):
        """New agent connected"""

        if mac not in i2c_ADDRESSES:
            print(f"Agent with MAc {mac} not in i2c_ADDRESSES (in config.py)!")
            return

        stim_addr, stim_chan, id = i2c_ADDRESSES[mac]

        agent = Agent(
            id=id,
            ip=host,
            port=port,
            mac=mac,
            stim_addr=stim_addr,
            stim_chan=stim_chan,
        )

        self.agents[host] = agent
        print(f" = new agent added (ESP {id})=")

    def collect(self):
        while True:
            try:
                (host, port), values_dict = self.sensor_q.get(block=False)
            except Empty:
                break

            if host not in self.agents:
                mac = values_dict["mac"]
                self.create_agent(mac, host, port)

            sensor_data = self.agents[host].sensor_data

            for dk in DATA_KEYS:
                if dk not in values_dict:
                    continue
                values = values_dict[dk]
                name = DATA_NAME_MAP[dk]

                max_hist = (
                    MAX_PLOT_SAMPLES if DATA_UNITS[dk] == "samples" else MAX_PLOT_FRAMES
                )

                sensor_data[name] = np.concatenate([sensor_data.get(name, []), values])[
                    -max_hist:
                ]

            # self.agents[host].sensor_data = sensor_data

        # plot_data = {
        #    "sensors": self.sensor_data,
        # }
        # self.plot_q.put(plot_data)

    def getFeatures(self):
        now = time()
        if now - self.last_feature_time < UPDATE_TIME:
            return

        self.last_feature_time = now

        target = self.getCenter()

        for host, agent in self.agents.items():
            agent = self.agents[host]
            features = dict()
            for name in DATA_NAME_MAP.values():
                buffer = agent.sensor_data[name]
                result = feature_extractors[name](buffer)
                for k, v in result.items():
                    f_name = name + ":" + k
                    features[f_name] = v

            # EMBED SENSORS
            x = self.makeFeatureVector(features)
            agent.feature_vectors.append(x)

            z = self.embedderNetwork(x)

            curr_state = np.concatenate([z, target])
            deterministic = False

            # MAKE ACTION
            # action `a` \in [-1, 1]^n
            if len(agent.output_vectors) > 0:
                # previous state of position and target...
                prev_state = np.concatenate([agent.map[-1], self.targets[-1]])
                # ...and previous actions...
                action = agent.output_vectors[-1]
                # ...led to this state z...
                # ...and so recieves the reward:
                reward = -np.linalg.norm(z - target)
                agent.hightlight = reward > -0.1

                self.translator.add(prev_state, action, reward, curr_state, False)
                deterministic = True

            a = self.translator.get_action(curr_state, deterministic)
            a = np.tanh(a * 5)

            agent.output_vectors.append(a)
            agent.map.append(z)

            self.postOutput(host, a)

        self.targets.append(target)

        # plot_data = {
        #     "map": self.map.copy(),
        #     "target": target,
        #     "feature_vectors": self.feature_vectors.copy(),
        #     "output_vectors": self.output_vectors.copy(),
        #     "embedder_losses": self.embedderNetwork.losses,
        #     "translator_losses": self.translator.losses,
        # }

        # self.plot_q.put(plot_data)

        print("-----------")

    def run(self):
        self.running.set()
        while self.running.is_set():
            try:
                self.collect()
                self.getFeatures()
                if time() >= self.last_checkpoint_time + CHECKPOINT_INTERVAL:
                    self.save()
                sleep(0.2)
            except KeyboardInterrupt:
                self.running.clear()

    def join(self, timeout=1):
        print("Translator join()")
        self.save()
        self.running.clear()
        super(Translator, self).join(timeout)

    def save(self):
        print("saving checkpoints")
        SAC_state_dicts = self.translator.save()
        embedder_state_dict = self.embedderNetwork.model.state_dict()

        data = {
            "sac": SAC_state_dicts,
            "embedder": embedder_state_dict,
        }

        fn = datetime.now().strftime("%Y%m%d_%H%M%S.pth")
        torch.save(data, os.path.join(CHECKPOINT_PATH, fn))

        self.last_checkpoint_time = time()

    def load(self, path: str):
        try:
            data = torch.load(path)
        except FileNotFoundError:
            print(f"{path} does not exist. Cannot load models.")
            return

        self.translator.load(data["sac"])
        self.embedderNetwork.model.load_state_dict(data["embedder"])

    def loadLatest(self):
        files = list(sorted(os.listdir(CHECKPOINT_PATH)))
        if len(files) == 0:
            print(f"No checkpoints found in {CHECKPOINT_PATH}")
            return

        latest_path = os.path.join(CHECKPOINT_PATH, files[-1])
        self.load(latest_path)


def post_request(url, headers, data, timeout=1.0):
    try:
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        res = session.post(url, json=data, headers=headers, timeout=timeout)
    except requests.exceptions.RequestException as e:
        print("POST request error to", url)
        print(e)
        return

    res.close()


def plot(plot_q: multiprocessing.Queue):
    pltr = MinPlotter(plot_q)

    try:
        ani = animation.FuncAnimation(
            pltr.fig,
            pltr.animate,
            interval=250,
            blit=True,
        )
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        ani.pause()
        print("plot KeyboardInterrupt")
        return


def handle_client(sensor_q, conn, addr):
    try:
        with conn:
            print("\n", "*" * 20)
            print(f"made connection to {addr}")
            print("*" * 20, "\n")
            data = b""
            while True:
                msg = conn.recv(BUFFERSIZE)
                data = data + msg
                data_decoded = msg.decode("ascii")
                if "#" not in data_decoded:
                    continue

                lines = data.decode("ascii").split("#")
                for line in lines[:-1]:
                    values = processBuffer(line)
                    for dk, v in values.items():
                        try:
                            name = DATA_NAME_MAP[dk]
                        except KeyError:
                            continue

                        if name not in processors:
                            continue

                        values[dk] = processors[name](np.array(v, dtype=float))

                    sensor_q.put((addr, values))
                data = bytes(lines[-1], "ascii")

    except Exception as e:
        print(f"[DEBUG] exception from {addr}")
        traceback.print_exc()
        print(e)

        conn.close()


def processBuffer(data):
    try:
        values = json.loads(data)
        return values
    except Exception as e:
        print(e)
        return {"data0": [], "data1": [], "data2": [], "data3": []}


def server(sensor_q):
    """
    Listens for new TCP connections and starts a new thread for each
    sensor ESP.
    """
    print("starting server thread...")

    threads = []

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, PORT))
        except OSError:
            print(f"## server port {PORT} already bound...")
            s.close()
            return

        s.listen()

        print(f"Server listening on {HOST}:{PORT}")

        try:
            while True:
                conn, addr = s.accept()

                t = threading.Thread(
                    target=handle_client,
                    args=(sensor_q, conn, addr),
                )
                t.start()
                threads.append(t)

        except KeyboardInterrupt:
            print("server KeyboardInterrupt")
            for t in threads:
                t.join(0.2)
            s.close()
            return


def lighthouse():
    """
    Listens for sensor ESPs UDP broadcasts and echos messages to estamblish
    Port and IP of server
    """
    print("starting lighthouse")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.bind(("", PORT))
            print("lighthouse started on", PORT)
        except OSError:
            print(f"#### lighthouse port {PORT} already bound...")
            s.close()
            return

        try:
            while True:
                msg, (host, port) = s.recvfrom(BUFFERSIZE)
                mac = msg.decode("utf-8")
                mac = mac.replace("\r", "")
                mac = mac.replace("\n", "")
                print(f"MAC Address found: {mac} from '{host}'")

                ##if mac not in i2c_ADDRESSES:
                ##    print(" === Unassigned device (not found in i2c_ADDRESSES)")

                ##mac_adderesses[host] = mac
                s.sendto(b"lighthouse", (host, port))

        except KeyboardInterrupt:
            print("lighthouse KeyboardInterrupt")
            return


def main():
    # raw sensor data is collected by 'handle_client' threads and sent
    # to the Translator over 'sensor_q'
    sensor_q = multiprocessing.Queue()

    # data to be plotted is added to the 'plot_q' by the Translator
    plot_q = multiprocessing.Queue()

    # Translator processes sensor data and makes stimulator outputs
    translator = Translator(
        sensor_q,
        plot_q,
        embedder_params,
        decision_params,
        translator_params,
        load_latest=args.load,
    )
    translator.start()

    # server process listens to new connections from ESPs
    server_process = multiprocessing.Process(None, server, args=(sensor_q,))
    server_process.start()

    # lighthouse_process echos ESPs broadcast messages to establish
    # connections
    lighthouse_process = multiprocessing.Process(None, lighthouse)
    lighthouse_process.start()

    # start plotting of data
    print("starting plot")
    plot(plot_q)

    print("finishing")

    server_process.join(0.2)
    lighthouse_process.join(0.2)
    translator.join(0.2)

    print("\nDONE")

    print(translator.is_alive())
    print(server_process.is_alive())
    print(lighthouse_process.is_alive())


if __name__ == "__main__":
    main()
