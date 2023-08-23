#!/bin/python

import os
import csv
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
from dataclasses import dataclass, field, asdict

import requests
from requests.adapters import Retry, HTTPAdapter

from dataclasses_json import dataclass_json, config

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from librosa import samples_to_frames, samples_to_time, frames_to_time

import data_processors as dp
from networks import LinearNetwork, VAENetwork, SACModel

import torch
import torch.nn as nn

from plotters import FullPlotter, MinPlotter, external_plotter

from config import (
    i2c_ADDRESSES,
    STIM_ADDR,
    TRACE_DIR,
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
parser.add_argument(
    "-t", 
    "--trace", 
    default=False, 
    required=False, 
    action="store_true", 
    help="save .csv files tracing agent's state"
)

HOST = "0.0.0.0"
PORT = 8080

SR = 512
FRAME_LENGTH = 512
HOP_LENGTH = 128
FR = samples_to_frames(SR, hop_length=HOP_LENGTH)

INPUT_MAX = 4096

BUFFERSIZE = 1024
MAX_PLOT_SAMPLES = 4000
MAX_PLOT_FRAMES = samples_to_frames(MAX_PLOT_SAMPLES, hop_length=HOP_LENGTH)

UPDATE_TIME = 4


processors = {
    "EEG": dp.ProcessorList(
        #dp.RecordProcessor(sr=SR, fn=f"eeg_raw_{SR}hz.txt"),
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


@dataclass_json
@dataclass
class Agent:
    ip: str
    port: int
    mac: str
    id: int
    map: deque = field(
        default_factory=lambda: deque(maxlen=8)
    )  # , metadata=config(encoder=list))
    active: bool = False
    highlight: bool = False
    stim_addr: int = 0x1A
    stim_chan: int = 1
    sensor_data: Dict = field(default_factory=dict)
    feature_vectors: deque = field(
        default=deque(maxlen=8), metadata=config(encoder=list)
    )
    output_vectors: deque = field(
        default=deque(maxlen=8), metadata=config(encoder=list)
    )

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
        save_trace: bool = False,
    ):

        super(Translator, self).__init__()
        self.sensor_q = sensor_q
        self.plot_q = plot_q

        self.last_feature_time = 0
        self.last_output_time = dict()

        self.last_checkpoint_time = 0
        self.last_checkpoint_time = time()

        self.save_trace = save_trace

        self.embedderNetwork = VAENetwork(embedder_params)
        self.translator = SACModel(**translator_params)
        #print("server: debug", self.translator.training_data.debug, id(self.translator.training_data.debug))

        self.targets = deque(maxlen=8)

        self.running = multiprocessing.Event()

        self.agents = dict()  # {host: agent}
        self.updated = False

        self.stim_addr = STIM_ADDR
        
        self.connections = dict()  # save IP addresses for debugging

        if load_latest:
            self.loadLatest()

        self.session_name = datetime.now().strftime("Session_%Y%m%d_%H%M%S")

        if self.save_trace:
            os.mkdir(os.path.join(TRACE_DIR, self.session_name))

        self.fieldnames = ["time"] + FEATURE_VECTOR_MAP + OUTPUT_VECTOR + ["map_x", "map_y", "active", "highlight"]
        #self.trace_writer = csv.DictWriter()

        print("Translator init done")
        #self.save()

    def getActiveAgents(self) -> Dict[str, Agent]:
        active = dict(
            [(host, agent) for host, agent in self.agents.items() if agent.active]
        )
        return active

    @staticmethod
    def makeFeatureVector(features) -> np.ndarray:
        x = [features.get(f, 0) for f in FEATURE_VECTOR_MAP]
        x = np.array(x)

        return x

    def getCenter(self) -> np.ndarray:
        """Returns the coordinate of the center of mass of all the points"""
        positions = []
        for host, agent in self.getActiveAgents().items():
            if len(agent.map) > 0:
                positions.append(agent.map[-1])

        if len(positions) <= 1:
            # either one agent or none, try last collective target
            if len(self.targets) > 0:
                return self.targets[-1]

            # else return center
            return np.array([0.0, 0.0])

        # multiple active agents, use mean positions
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

        url = f"http://{self.stim_addr}/stim"
        #print(url)
        headers = {"Content-Type": "application/json"}
        data = {
            "channel": agent.stim_chan,
            "addr": agent.stim_addr,
        }

        host_url = f"http://{host}/update"
        update_data = {
            "highlight": agent.highlight,
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
        
        self.connections[id] = (mac, host, port)
        with open("./connections.json", "w") as f:
            json.dump(self.connections, f, sort_keys=True, indent=4)

        self.agents[host] = agent
        self.updated = True
        print(f"** new agent added (ESP {id}) **")

        if self.save_trace:
            with open(os.path.join(TRACE_DIR, self.session_name, f'ESP{agent.id}.csv'), "w") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
                writer.writeheader()

    def collect(self):
        while True:
            try:
                (host, port), values_dict = self.sensor_q.get(block=False)
            except Empty:
                break

            #print(values_dict)

            if "stim_addr" in values_dict:
                self.stim_addr = values_dict["stim_addr"]
                continue

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

            if "active" in values_dict:
                # print(host, values_dict["active"])
                self.agents[host].active = values_dict["active"]

            self.updated = True

    def getFeatures(self):
        now = time()
        if now - self.last_feature_time < UPDATE_TIME:
            return

        self.last_feature_time = now

        print("getFeatures")
        target = self.getCenter()

        for host, agent in self.getActiveAgents().items():
            features = dict()
            print(host)
            try:
                for name in DATA_NAME_MAP.values():
                    buffer = agent.sensor_data[name]
                    result = feature_extractors[name](buffer)
                    for k, v in result.items():
                        f_name = name + ":" + k
                        features[f_name] = v
            except KeyError:
                continue

            # EMBED SENSORS
            x = self.makeFeatureVector(features)
            agent.feature_vectors.append(x)

            z = self.embedderNetwork(x)

            curr_state = np.concatenate([z, target])
            deterministic = False

            # MAKE ACTION
            # action `a` \in [-1, 1]^n
            if len(agent.output_vectors) > 0 and len(self.targets) > 0:
                # previous state of position and target...
                prev_state = np.concatenate([agent.map[-1], self.targets[-1]])
                # ...and previous actions...
                action = agent.output_vectors[-1]
                # ...led to this state z...
                # ...and so recieves the reward:
                reward = -np.linalg.norm(z - target)
                agent.highlight = bool(reward > -0.2)

                self.translator.add(prev_state, action, reward, curr_state, False)
                deterministic = True

            a = self.translator.get_action(curr_state, deterministic)
            a = np.tanh(a * 5)

            agent.output_vectors.append(a)
            agent.map.append(z)

            self.agents[host] = agent
            self.postOutput(host, a)

            if self.save_trace:
                with open(os.path.join(TRACE_DIR, self.session_name, f'ESP{agent.id}.csv'), "a") as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')

                    line = dict(features)
                    line["time"] = int(time())
                    line["active"] = agent.active
                    line["highlight"] = agent.highlight
                    for i, name in enumerate(OUTPUT_VECTOR):
                        line[name] = a[i]
                    line["map_x"] = z[0]
                    line["map_y"] = z[1]

                    writer.writerow(line)

        self.targets.append(target)

    def updatePlotQueue(self):
        if self.updated:
            self.plot_q.put({
                "agents": self.agents, 
                "center": self.targets[-1],
                "embedding_losses": self.embedderNetwork.losses,
                "translator_losses": self.translator.losses,
            })
            self.updated = False

    def run(self):
        self.running.set()
        while self.running.is_set():
            try:
                self.collect()
                self.getFeatures()
                self.updatePlotQueue()
                hour = datetime.now().hour
                if (time() >= self.last_checkpoint_time + CHECKPOINT_INTERVAL) and hour > 10 and hour < 20:
                    print("checkpointing")
                    self.save()
                sleep(0.2)
            except KeyboardInterrupt:
                self.save()
                self.running.clear()

    def join(self, timeout=1):
        print("Translator join()")
        self.running.clear()
        super(Translator, self).join(timeout)

    def save(self):
        print(f"saving checkpoints... on thread {threading.get_ident()}")
        training_data = self.translator.training_data.to_dict()

        SAC_state_dicts = self.translator.save()
        embedder_state_dict = self.embedderNetwork.model.state_dict()

        data = {
            "sac": SAC_state_dicts,
            "embedder": embedder_state_dict,
            "buffer": training_data,
        }

        fn = datetime.now().strftime("%Y%m%d_%H%M%S.pth")
        torch.save(data, os.path.join(CHECKPOINT_PATH, fn))

        self.last_checkpoint_time = time()
        print(f"done (saved to {os.path.join(CHECKPOINT_PATH, fn)})\n")

    def load(self, path: str):
        print("load()")
        try:
            data = torch.load(path)
        except FileNotFoundError:
            print(f"{path} does not exist. Cannot load models.")
            return

        self.translator.load(data["sac"])
        self.embedderNetwork.model.load_state_dict(data["embedder"])
        if "buffer" in data:
            print("loading training data")
            self.translator.training_data.from_dict(data["buffer"])

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
        # print("POST request error to", url)
        # print(e)
        return

    res.close()


def plot(plot_q: multiprocessing.Queue):
    #pltr = FullPlotter(plot_q)

    try:
        # ani = animation.FuncAnimation(
        #     pltr.fig,
        #     pltr.animate,
        #     interval=250,
        #     blit=True,
        # )
        # plt.tight_layout()
        # plt.show()

        # t = threading.Thread(
        #    target=external_plotter,
        #    args=(plot_q),
        # )
        # t.start()

        external_plotter(plot_q)

    except KeyboardInterrupt:
        # ani.pause()
        print("plot KeyboardInterrupt")
        return


def handle_client(sensor_q, conn, addr):
    try:
        with conn:
            print("\n", "*" * 20)
            print(f" made connection to {addr}")
            print("", "*" * 20, "\n")
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

                        if name == "GSR" and len(values["data3"]) > 0:
                            active = bool(np.mean(values["data3"]) > 10)
                            values["active"] = active

                        values[dk] = processors[name](np.array(v, dtype=float))

                    sensor_q.put((addr, values))
                data = bytes(lines[-1], "ascii")

    except Exception as e:
        print(f"[DEBUG] exception from {addr}")
        # traceback.print_exc()
        print(e)

    conn.close()
    print("! handle_client disconnected!")


def processBuffer(data):
    try:
        values = json.loads(data)
        return values
    except Exception as e:
        print(e)
        #print(data)
        return {"data0": [], "data1": [], "data2": [], "data3": [], "active": False}


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
        
        connections = []
        
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


def lighthouse(sensor_q):
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

                if mac == "stimulator":
                    s.sendto(b"lighthouse", (host, port))
                    STIM_ADDR = host
                    #print("***** STUMULATOR IP FOUND ******")
                    #print(f"****** {host} *****")

                    sensor_q.put(((host, port), {"stim_addr": host}))
                    continue

                print(f"MAC Address found: {mac} from '{host}'")

#                 if mac not in i2c_ADDRESSES:
#                     print(" === Unassigned device (not found in i2c_ADDRESSES)")

#                 mac_adderesses[host] = mac
                s.sendto(b"lighthouse", (host, port))

        except KeyboardInterrupt:
            print("lighthouse KeyboardInterrupt")
            return


def main():    
    args = parser.parse_args()
    if args.port:
        PORT = args.port
    
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
        save_trace=args.trace,
    )
    translator.start()

    # server process listens to new connections from ESPs
    server_process = multiprocessing.Process(None, server, args=(sensor_q,))
    server_process.start()

    # lighthouse_process echos ESPs broadcast messages to establish
    # connections
    lighthouse_process = multiprocessing.Process(None, lighthouse, args=(sensor_q,))
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
