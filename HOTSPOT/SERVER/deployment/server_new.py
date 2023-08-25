#!/bin/python

import os
import csv
import json
import struct
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
from http.server import BaseHTTPRequestHandler, HTTPServer

from apscheduler.schedulers.background import BackgroundScheduler

import numpy as np
from librosa import samples_to_frames, samples_to_time, frames_to_time

import torch

import data_processors as dp
from networks import LinearNetwork, VAENetwork, SACModel

from config import (
    HOST,
    PORT,
    STATIONS,
    TRACE_DIR,
    UPDATE_TIME,
    CHECKPOINT_INTERVAL,
    CHECKPOINT_PATH,
    DATA_KEYS,
    DATA_UNITS,
    DATA_NAME_MAP,
    FEATURE_VECTOR_MAP,
    OUTPUT_VECTOR,
    OUTPUT_VECTOR_RANGES,
    MCAST_GRP,
    MCAST_PORT,
)

from esp_types import ESP, Agent

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", 
    "--port",
    metavar="port",
    type=int, 
    default=8080, 
    required=False,
    help="port to start HTTP listening server on",
)
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
    help="save .csv files tracing agent's state",
)
parser.add_argument(
    "--host", 
    metavar="host",
    type=str,
    default="",
    required=False, 
    help="host IP to start HTTP listening server on",
)


SR = 512
FRAME_LENGTH = 512
HOP_LENGTH = 128
FR = samples_to_frames(SR, hop_length=HOP_LENGTH)

INPUT_MAX = 4096

BUFFERSIZE = 1024
MAX_PLOT_SAMPLES = 4000
MAX_PLOT_FRAMES = samples_to_frames(MAX_PLOT_SAMPLES, hop_length=HOP_LENGTH)


processors = {
    "EEG": dp.ProcessorList(
        dp.NormaliseProcessor(
            sr=SR, inrange=(0, INPUT_MAX), max_hist=MAX_PLOT_SAMPLES * 16
        ),
        dp.FilterProcessor(sr=SR, Wn=(25, 45), btype="bandpass"),
        dp.RMSProcessor(sr=SR, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH),
        dp.RescaleProcessor(sr=FR, in_range=(0, 1), out_range=(-1, 1)),
    ),
    "EOG": dp.ProcessorList(
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


class Translator(multiprocessing.Process):
    def __init__(
        self,
        msg_q: multiprocessing.Queue,
        plot_q: multiprocessing.Queue,
        embedder_params: Dict,
        decision_params: Dict,
        translator_params: Dict,
        mcast_socket: socket.socket = None,
        load_latest: bool = False,
        save_trace: bool = False,
    ):

        super(Translator, self).__init__()
        self.msg_q = msg_q
        self.plot_q = plot_q

        self.save_trace = save_trace

        self.embedderNetwork = VAENetwork(embedder_params)
        self.translator = SACModel(**translator_params)

        self.mcast_socket = mcast_socket

        self.targets = deque(maxlen=8)

        self.running = multiprocessing.Event()

        self.agents = dict()  # {host: agent}
        self.updated = False
        
        self.connections = dict()  # save IP addresses for debugging

        if load_latest:
            self.loadLatest()

        self.session_name = datetime.now().strftime("Session_%Y%m%d_%H%M%S")

        if self.save_trace:
            os.mkdir(os.path.join(TRACE_DIR, self.session_name))

        self.fieldnames = ["time"] + FEATURE_VECTOR_MAP + OUTPUT_VECTOR + ["map_x", "map_y", "active", "highlight"]

        print("Translator init done")

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

    def output(self):
        data = dict()

        for _, agent in self.agents.items():
            parameters = dict()
            try:
                y = agent.output_vectors[-1]
            except (IndexError, KeyError):
                continue

            for i, name in enumerate(OUTPUT_VECTOR):
                if name in ("temp1", "temp2", "highlight"):
                    parameters[name.lower()] = val
                else:
                    val = np.clip(y[i], -1, 1)
                    r = OUTPUT_VECTOR_RANGES[name]
                    out = int((val / 2 + 0.5) * (r[1] - r[0]) + r[0])
                    parameters[name.lower()] = out

            data[str(agent.station)] = parameters

        if len(data) > 0:
            self.broadcast(data)


    def broadcast(self, data):
        if self.mcast_socket is None:
            return

        json_data = json.dumps(data)
        sent = self.mcast_socket.sendto(bytes(json_data, 'utf-8'), (MCAST_GRP, MCAST_PORT))

    def create_agent(self, mac: str, host: str, port: int):
        """New agent connected"""

        if mac not in STATIONS:
            print(f"Agent with MAC {mac} not in STATIONS (in config.py)!")
            return

        station, id_, esp_type = STATIONS[mac]

        agent = Agent(
            id=id_,
            ip=host,
            port=port,
            mac=mac,
            station=station,
            esp_type=esp_type,
        )
        
        self.connections[id_] = (mac, host, port)
        with open("./connections.json", "w") as f:
            json.dump(self.connections, f, sort_keys=True, indent=4)

        self.agents[host] = agent
        self.updated = True
        print(f"** new agent added (ESP {id_}, {esp_type.name}, {mac}, {host}:{port}) **")

        data = dict()
        data[mac] = {"type": "whoami", "station": station}
        self.broadcast(data)

        if self.save_trace:
            with open(os.path.join(TRACE_DIR, self.session_name, f'ESP{agent.id}.csv'), "w") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
                writer.writeheader()

    def collect(self):
        while True:
            try:
                (host, port), msg = self.msg_q.get(block=False)
            except Empty:
                break

            msg_type = msg.get("type", "unknown")

            if msg_type == "ping":
                self.handlePingMessage(msg, host, port)
 
            elif msg_type == "sensor":
                self.handleSensorMessage(msg, host, port)
                self.updated = True

            elif msg_type == "whoami":
                pass

            elif msg_type == "checkpoint":
                self.save()

            elif msg_type == "features":
                self.getFeatures()

            elif msg_type == "output":
                self.output()

            else:
                print(f"Unkown message from {host}, MAC {self.agents.get(host, 'unregistered')}")


    def handlePingMessage(self, msg, host, port):
        if host not in self.agents:
            mac = msg["mac"]
            self.create_agent(mac, host, port)

        self.agents[host].last_ping_time = time()

    def handleSensorMessage(self, msg, host, port):
        if host not in self.agents:
            print("not in self.agents")
            return

        sensor_data = self.agents[host].sensor_data

        for dk in DATA_KEYS:
            if dk not in msg:
                continue
            values = msg[dk]
            name = DATA_NAME_MAP[dk]

            max_hist = (
                MAX_PLOT_SAMPLES if DATA_UNITS[dk] == "samples" else MAX_PLOT_FRAMES
            )

            sensor_data[name] = np.concatenate([sensor_data.get(name, []), values])[
                -max_hist:
            ]

        if "active" in msg:
            self.agents[host].active = msg["active"]


    def getFeatures(self):
        target = self.getCenter()

        for host, agent in self.getActiveAgents().items():
            features = dict()
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
            # self.postOutput(host, a)

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
        if self.updated and self.plot_q is not None:
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
                self.updatePlotQueue() 
                # sleep(0.2)
            except KeyboardInterrupt:
                self.save()
                self.running.clear()

    def join(self, timeout=1):
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
        print(f"done saving checkpoints ({os.path.join(CHECKPOINT_PATH, fn)})\n")

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


def processBuffer(data):
    try:
        values = json.loads(data)
        return values
    except Exception as e:
        print(e)
        return {"data0": [], "data1": [], "data2": [], "data3": [], "active": False, "type": "sensor"}


def makeHTTPServer(msg_q):
    class HH_HttpServer(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.msg_q = msg_q
            super(HH_HttpServer, self).__init__(*args, **kwargs)

        def _set_response(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            json_data = post_data.decode('utf-8')

            addr = self.client_address
            values = processBuffer(json_data)
            if "server" not in values:
                self._set_response()
                return

            values = values["server"]

            for dk, v in values.items():
                try:
                    name = DATA_NAME_MAP[dk]
                except KeyError:
                    continue

                if name not in processors:
                    continue

                if name == "GSR" and len(values["data3"]) > 0:
                    active = bool(np.mean(values["data3"]) > 8)
                    values["active"] = active

                values[dk] = processors[name](np.array(v, dtype=float))

            msg_q.put((addr, values))
            self._set_response()

        def log_request(self, code='-', size='-'): 
            return
    
    return HH_HttpServer


def lighthouse(mcast_socket):
    sent = mcast_socket.sendto(
        bytes("{\"all\": {\"type\": \"lighthouse\", \"port\": " + str(PORT) + "}}", 'ascii'), 
        (MCAST_GRP, MCAST_PORT)
    )
        
def multicast_listener(mcast_socket, msg_q):
    while True:
        try:
            data, addr = mcast_socket.recvfrom(1024)
        except socket.timeout:
            continue
        except KeyboardInterrupt:
            return
        else:
            try:
                json_data = json.loads(data.decode('utf-8'))
                print(json_data)
            except json.decoder.JSONDecodeError:
                print("error parsing", data.decode('utf-8'))

            if "server" in json_data:
                msg_q.put((addr, json_data["server"]))



def getInterfaceIP(interface="wlan0"):
    from subprocess import check_output

    output = check_output(["ip", "-j", "addr", "show", interface])
    d = json.loads(output)

    try:
        host = d[0]['addr_info'][0]['local']
    except (IndexError, KeyError):
        print(f"** Cannot establish interface {interface} IP address. Defaulting to localhost")
        host = '127.0.0.1'

    return host


def main():
    global HOST, PORT

    args = parser.parse_args()
    if args.port:
        PORT = args.port

    if args.host:
        HOST = args.host
    elif HOST == '':
        HOST = getInterfaceIP()

    multicast_group = (
        MCAST_GRP,
        MCAST_PORT
    )

    mcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mcast_socket.settimeout(0.2)
    mcast_socket.bind(multicast_group)

    print()
    print("="*50)
    print(f"  Multicast group:   {MCAST_GRP}:{MCAST_PORT}")
    print("="*50)
    print()

    group = socket.inet_aton(multicast_group[0])
    mreq = struct.pack('4sL', group, socket.INADDR_ANY)
    mcast_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    ttl = struct.pack('b', 1)
    mcast_socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
    
    msg_q = multiprocessing.Queue()

    def add_message(msg):
        msg_q.put(((HOST, PORT), msg))

    scheduler = BackgroundScheduler()
    scheduler.add_job(lighthouse, 'interval', args=(mcast_socket,), seconds=10)
    scheduler.add_job(add_message, 'interval', args=({"type": "checkpoint"},), seconds=CHECKPOINT_INTERVAL)
    scheduler.add_job(add_message, 'interval', args=({"type": "features"},), seconds=UPDATE_TIME)
    scheduler.add_job(add_message, 'interval', args=({"type": "output"},), seconds=UPDATE_TIME)

    translator = Translator(
        msg_q,
        None,
        embedder_params,
        decision_params,
        translator_params,
        mcast_socket=mcast_socket,
        load_latest=args.load,
        save_trace=args.trace,
    )

    handler_class = makeHTTPServer(msg_q)
    httpd = HTTPServer((HOST, PORT), handler_class)
    http_thread = threading.Thread(target=httpd.serve_forever)
    http_thread.daemon = True

    print()
    print("="*50)
    print(f"  HTTP Server:       {HOST}:{PORT}")
    print("="*50)
    print()

    translator.start()
    scheduler.start()
    http_thread.start()

    print()
    print("READY")
    print()
    print("Received messages:")
    print("-"*50)

    try:
        multicast_listener(mcast_socket, msg_q)
    except KeyboardInterrupt:
        pass

    print("\nCLOSING")

    httpd.shutdown()
    scheduler.shutdown()
    translator.join()

    print("\nDONE")


if __name__ == "__main__":
    main()