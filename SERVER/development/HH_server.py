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

import requests
from requests.adapters import Retry, HTTPAdapter

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from librosa import samples_to_frames, samples_to_time, frames_to_time

import data_processors as dp
from networks import LinearNetwork, VAENetwork, SACModel

import torch.nn as nn

from config import i2c_ADDRESSES, STIM_ADDR, CHECKPOINT_INTERVAL, CHECKPOINT_PATH

matplotlib.rcParams["toolbar"] = "None"
plt.style.use("dark_background")
matplotlib.rcParams["axes.edgecolor"] = "#AAA"
matplotlib.rcParams["xtick.color"] = "#AAA"
matplotlib.rcParams["ytick.color"] = "#AAA"
matplotlib.rcParams["text.color"] = "#AAA"

parser = argparse.ArgumentParser()
parser.add_argument("port", type=int)
parser.add_argument("load", action="store_true")

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


DATA_KEYS = ["data0", "data1", "data2", "data3"]
DATA_NAME_MAP = {
    "data0": "EEG",
    "data1": "EOG",
    "data2": "heart",
    "data3": "GSR",
}

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


DATA_UNITS = {
    "data0": "frames",
    "data1": "samples",
    "data2": "samples",
    "data3": "samples",
}

AX_LIMS = {
    "EEG": (-1.1, 1.1),
    "EOG": (-1.1, 1.1),
    "GSR": (-1.1, 1.1),
    "heart": (-1.1, 1.1),
}

CMAP = matplotlib.cm.get_cmap("rainbow")

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

OUTPUT_VECTOR = ["ampl", "freq", "durn", "idly", "temp1", "temp2", "indicate"]

OUTPUT_VECTOR_RANGES = {
    "ampl": [3, 12],
    "freq": [1, 100],
    "durn": [0, 2000],
    "idly": [0, 255],
    "temp1": [-1, 1],
    "temp2": [-1, 1],
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
    "state_dim": embedder_params["latent_size"]
    * 2,  # current location + target location
    "action_dim": len(OUTPUT_VECTOR),
    "gamma": 0.99,
    "hid_shape": (12, 12),
    "a_lr": 0.0001,
    "c_lr": 0.0001,
    "batch_size": 1,
    "alpha": 0.12,
    "adaptive_alpha": False,
}


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

        self.sensor_data = dict()
        self.feature_vectors = dict()
        self.output_vectors = dict()
        self.map = dict()
        self.targets = deque(maxlen=8)

        self.running = multiprocessing.Event()

        self.connections = dict()
        self.mac_adderesses = dict()

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
        for host, hist in self.map.items():
            positions.append(hist[-1])

        return np.mean(positions, axis=0)

    def postOutput(self, host: str, y: np.ndarray, reward: float = -10):
        if host not in self.last_output_time:
            self.last_output_time[host] = 0

        if host not in self.connections:
            print(host, "not in connections")
            return

        now = time()
        if now - self.last_output_time[host] < 4.0:
            return

        self.last_output_time[host] = now

        try:
            mac = self.mac_adderesses[host]
        except KeyError:
            print(
                host,
                "unknown / not registered host (has not registered with this server, restart that ESP?)",
            )
            return

        try:
            addr, chan, _ = i2c_ADDRESSES[mac]
        except KeyError:
            print(mac, "not assigned! (make sure it is specified in `config.py`)")
            return

        url = f"http://{STIM_ADDR}/stim"
        headers = {"Content-Type": "application/json"}
        data = {
            "channel": chan,
            "addr": addr,
        }

        temp_url = f"http://{host}/temp"
        temp_data = {}

        indicate_url = f"http://{host}/indicate"
        indicate_data = {"indicate": False}
        if -reward < 0.1:
            indicate_data["indicate"] = True

        for i, name in enumerate(OUTPUT_VECTOR):
            val = np.clip(y[i], -1, 1)

            if name in ("temp1", "temp2"):
                temp_data[name.lower()] = val
            else:
                r = OUTPUT_VECTOR_RANGES[name]
                out = int((val / 2 + 0.5) * (r[1] - r[0]) + r[0])
                data[name.lower()] = out

        # print(data)
        t1 = threading.Thread(
            target=post_request,
            args=(url, headers, data, 0.5),
        )
        t1.start()

        # print(temp_data)
        t2 = threading.Thread(
            target=post_request,
            args=(temp_url, headers, temp_data, 2.0),
        )
        t2.start()

        print(indicate_data)
        t3 = threading.Thread(
            target=post_request,
            args=(indicate_url, headers, indicate_data, 0.5),
        )
        t3.start()

    def collect(self):
        while True:
            try:
                (host, port), values_dict = self.sensor_q.get(block=False)
            except Empty:
                break

            if "mac" in values_dict:
                mac = values_dict["mac"]
                if host not in self.mac_adderesses:
                    self.mac_adderesses[host] = mac

            if host not in self.connections:
                self.connections[host] = len(self.connections)

            if host not in self.sensor_data:
                self.sensor_data[host] = dict()
                for dk in DATA_KEYS:
                    name = DATA_NAME_MAP[dk]
                    if DATA_UNITS[dk] == "samples":
                        self.sensor_data[host][name] = np.zeros(MAX_PLOT_SAMPLES)
                    else:
                        self.sensor_data[host][name] = np.zeros(MAX_PLOT_FRAMES)

            for dk in DATA_KEYS:
                if dk not in values_dict:
                    continue
                values = values_dict[dk]
                name = DATA_NAME_MAP[dk]
                if DATA_UNITS[dk] == "samples":
                    max_hist = MAX_PLOT_SAMPLES
                else:
                    max_hist = MAX_PLOT_FRAMES
                self.sensor_data[host][name] = np.concatenate(
                    [self.sensor_data[host][name], values]
                )[-max_hist:]

        plot_data = {
            "sensors": self.sensor_data,
        }
        self.plot_q.put(plot_data)

    def getFeatures(self):
        now = time()
        if now - self.last_feature_time < UPDATE_TIME:
            return

        self.last_feature_time = now

        target = self.getCenter()
        self.targets.append(target)

        for host in self.sensor_data.keys():
            features = dict()
            for name in DATA_NAME_MAP.values():
                buffer = self.sensor_data[host][name]
                result = feature_extractors[name](buffer)
                for k, v in result.items():
                    f_name = name + ":" + k
                    features[f_name] = v

            # EMBED SENSORS
            x = self.makeFeatureVector(features)
            if host not in self.feature_vectors:
                self.feature_vectors[host] = deque(maxlen=8)
            self.feature_vectors[host].append(x)

            z = self.embedderNetwork(x)
            if host not in self.map:
                self.map[host] = deque(maxlen=8)
            self.map[host].append(z)

            # MAKE ACTION
            # action `a` \in [-1, 1]^n
            if len(self.map[host]) > 2:
                # previous state of position and target...
                state = np.concatenate(self.map[host][-2], self.target[-2])
                # ...and previous actions...
                action = self.output_vectors[host][-2]
                # ...led to this state z...
                # ...and so recieves the reward:
                reward = -np.linalg.norm(z - target)

                self.translator.add(state, action, reward, z, False)

            # get next action
            deterministic = False
            if host in self.output_vectors and len(self.output_vectors[host]) > 2:
                deterministic = True
            a = self.translator.get_action(z, deterministic)
            # amplify output:
            a = np.tanh(a * 5)

            # y = self.interpretterNetwork(z)[0]
            self.postOutput(host, a, reward)

            if host not in self.output_vectors:
                self.output_vectors[host] = deque(maxlen=8)
            self.output_vectors[host].append(a)

        plot_data = {
            "map": self.map.copy(),
            "target": target,
            "feature_vectors": self.feature_vectors.copy(),
            "output_vectors": self.output_vectors.copy(),
            "embedder_losses": self.embedderNetwork.losses,
            "translator_losses": self.translator.losses,
        }

        self.plot_q.put(plot_data)
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
        embedder_state_dict = self.embedderNetwork.state_dict()

        data = {
            "sac": SAC_state_dicts,
            "embedder": embedder_state_dict,
        }

        fn = datetime.now().strftime("%Y%m%d_%H%M%S.pth")
        torch.save(data, os.path.join([CHECKPOINT_PATH, fn]))

        self.last_checkpoint_time = time()

    def load(self, path: str):
        try:
            data = torch.load(path)
        except FileNotFoundError:
            print(f"{path} does not exist. Cannot load models.")
            return

        self.translator.load(data["sac"])
        self.embedderNetwork.load_state_dict(data["embedder"])

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


class Plotter:
    def __init__(self, plot_q, minimal: bool = False):
        self.plot_q = plot_q

        self.fig = plt.figure()
        self.gs = GridSpec(4, 3, width_ratios=[1, 2, 2])
        self.gs.tight_layout(self.fig)

        self.sensor_axes = [
            self.fig.add_subplot(self.gs[0, 0]),
            self.fig.add_subplot(self.gs[1, 0]),
            self.fig.add_subplot(self.gs[2, 0]),
            self.fig.add_subplot(self.gs[3, 0]),
        ]

        for dk, ax in zip(DATA_KEYS, self.sensor_axes):
            name = DATA_NAME_MAP[dk]
            ax.set_ylim(*AX_LIMS[name])
            ax.set_xlim(-4, 0)
            ax.set_title(name)

        self.map_ax = self.fig.add_subplot(self.gs[:2, 1:])
        self.embed_loss_ax = self.fig.add_subplot(self.gs[2, 1])
        self.trans_loss_ax = self.fig.add_subplot(self.gs[2, 2])
        self.feat_vector_ax = self.fig.add_subplot(self.gs[3, 1])
        self.out_vector_ax = self.fig.add_subplot(self.gs[3, 2])

        self.map_ax.set_title("map")
        self.map_ax.set_ylim(-1.1, 1.1)
        self.map_ax.set_xlim(-1.1, 1.1)
        self.map_ax.set_xticks([])
        self.map_ax.set_yticks([])

        self.embed_loss_ax.set_title("embedding loss")
        self.embed_loss_ax.set_xlim(0, 16)
        self.embed_loss_ax.set_xticks([])
        self.trans_loss_ax.set_title("translator loss")
        self.trans_loss_ax.set_xlim(0, 16)
        self.trans_loss_ax.set_xticks([])

        self.feat_vector_ax.set_title("feature vector")
        self.feat_vector_ax.set_xticks([])
        self.feat_vector_ax.set_yticks(np.arange(len(FEATURE_VECTOR_MAP)))
        self.feat_vector_ax.set_yticklabels(FEATURE_VECTOR_MAP)

        self.out_vector_ax.set_title("output vector")
        self.out_vector_ax.set_xticks([])
        self.out_vector_ax.set_yticks(np.arange(len(OUTPUT_VECTOR)))
        self.out_vector_ax.set_yticklabels(OUTPUT_VECTOR)

        self.plot_data = {
            "sensors": dict(),
            "map": dict(),
            "feature_vectors": dict(),
            "output_vectors": dict(),
            "embedder_losses": [],
            "translator_losses": {
                "a_losses": [],
                "q_losses": [],
            },
        }

        self.lines = self.plot_data.copy()

        em_l = Line2D([], [], color="white")
        self.lines["embedder_losses"] = em_l
        self.embed_loss_ax.add_line(em_l)

        a_l = Line2D([], [], color="orange")
        self.lines["translator_losses"]["a_losses"] = a_l
        self.trans_loss_ax.add_line(a_l)
        q_l = Line2D([], [], color="aqua")
        self.lines["translator_losses"]["q_losses"] = q_l
        self.trans_loss_ax.add_line(q_l)

        feature_mat = np.zeros((1, len(FEATURE_VECTOR_MAP)))
        f_mat_show = self.feat_vector_ax.imshow(feature_mat.T, vmin=0, vmax=100)
        # self.feat_vector_ax.set_xlim(0, 8)
        self.lines["feature_vectors"] = f_mat_show

        output_mat = np.zeros((1, len(OUTPUT_VECTOR_RANGES)))
        o_mat_show = self.out_vector_ax.imshow(output_mat.T, vmin=-1, vmax=1)
        # self.out_vector_ax.set_xlim(0, 8)
        self.lines["output_vectors"] = o_mat_show

        self.colors = dict()

    def animate(self, num: int) -> List[Line2D]:
        """
        plot_data:
          - 'sensors'
              - <host>
                  - 'EEG' = np.ndarray
                  - 'EOG' = np.ndarray
                  - 'GSR' = np.ndarray
                  - 'heart' = np.ndarray
          - 'map'
              - <host> = list
          - 'feature_vectors'
              - <host> = list
          - 'output_vectors'
              - <host> = list
          - 'training_losses'
              - 'embedding' = list
              - 'intrepretter' = list
              - 'translator'
                - 'a_loss' = list
                - 'q_loss' = list
        """

        all_lines: List[Line2D] = []
        updated = False

        while True:
            try:
                plot_data = self.plot_q.get(block=False)
                self.plot_data.update(plot_data)
                updated = True
            except Empty:
                break

        if not updated:
            return all_lines

        # --------------- SENSORS ----------------
        for dk, ax in zip(DATA_KEYS, self.sensor_axes):
            # if "sensors" not in plot_data:
            #    break

            name = DATA_NAME_MAP[dk]
            for host in self.plot_data["sensors"].keys():
                buffers = self.plot_data["sensors"][host][name]

                t = t_f
                if DATA_UNITS[dk] == "samples":
                    t = t_sr

                if host not in self.colors:
                    c = CMAP(len(self.colors) / 8)
                    self.colors[host] = c

                if host not in self.lines["sensors"]:
                    self.lines["sensors"][host] = dict()

                if name not in self.lines["sensors"][host]:
                    line = Line2D(
                        t,
                        buffers,
                        linewidth=1,
                        alpha=0.5,
                        color=self.colors[host],
                    )
                    self.lines["sensors"][host][name] = line
                    ax.add_line(line)
                else:
                    self.lines["sensors"][host][name].set_data(t, buffers)

                all_lines.append(self.lines["sensors"][host][name])

        # --------------- MAP ----------------
        for host in self.plot_data["map"].keys():
            # if "map" not in plot_data:
            #    break

            m = np.stack(self.plot_data["map"][host])
            xs = m[:, 0]
            ys = m[:, 1]

            if host not in self.lines["map"]:
                line = Line2D(xs, ys, color=self.colors[host])
                self.lines["map"][host] = line
                self.map_ax.add_line(line)
            else:
                self.lines["map"][host].set_data(xs, ys)

            all_lines.append(self.lines["map"][host])

        # --------------- FEATURES ----------------
        # if "feature_vectors" in plot_data:
        fvs = self.plot_data["feature_vectors"]
        feature_mat = np.zeros((8, len(FEATURE_VECTOR_MAP)))
        for i, host in enumerate(self.colors.keys()):
            if host in fvs and len(fvs[host]) > 0:
                fv = fvs[host][-1]
                feature_mat[i] = fv

        if len(feature_mat):
            self.feat_vector_ax.matshow(feature_mat.T, vmin=-1, vmax=20)

        # if "output_vectors" in plot_data:
        ovs = self.plot_data["output_vectors"]
        output_mat = np.zeros((8, len(OUTPUT_VECTOR)))
        for i, host in enumerate(self.colors.keys()):
            if host in fvs and len(ovs[host]) > 0:
                ov = ovs[host][-1]
                output_mat[i] = ov

        if len(output_mat):
            # self.lines["output_vectors"].set_data(output_mat.T)
            # self.out_vector_ax.set_xlim(0, 8)
            self.out_vector_ax.matshow(output_mat.T, vmin=-1, vmax=1)

        # --------------- LOSSES ----------------
        # if "embedder_losses" in plot_data:
        em_losses = self.plot_data["embedder_losses"]
        if em_losses:
            self.lines["embedder_losses"].set_data(
                np.arange(len(em_losses)),
                em_losses,
            )
            all_lines.append(self.lines["embedder_losses"])
            self.embed_loss_ax.set_ylim(0, np.max(em_losses) * 1.1)

        # if "translator_losses" in plot_data:
        a_losses = self.plot_data["translator_losses"]["a_losses"]
        q_losses = self.plot_data["translator_losses"]["q_losses"]

        if a_losses:
            self.lines["translator_losses"]["a_losses"].set_data(
                np.arange(len(a_losses)), a_losses
            )
            all_lines.append(self.lines["translator_losses"]["a_losses"])
            self.lines["translator_losses"]["q_losses"].set_data(
                np.arange(len(q_losses)), q_losses
            )
            all_lines.append(self.lines["translator_losses"]["q_losses"])
            upper = max(np.max(a_losses), np.max(q_losses))
            lower = min(np.min(a_losses), np.min(q_losses))
            self.trans_loss_ax.set_ylim(
                lower - abs(0.1 * lower),
                upper + abs(0.1 * upper),
            )

        self.fig.canvas.draw()

        return all_lines


def plot(plot_q: multiprocessing.Queue):
    pltr = Plotter(plot_q)

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


def handle_client(q, conn, addr):
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

                    q.put((addr, values))
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
