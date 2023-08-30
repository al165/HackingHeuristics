import os
import csv
import json
import socket
import threading
import multiprocessing
from queue import Empty
from time import time, sleep
from datetime import datetime
from collections import deque
from typing import Dict, Tuple, List

import numpy as np
from torch import save, load

import data_processors as dp
from esp_types import Agent, ESP
from networks import LinearNetwork, VAENetwork, SACModel


from config import (
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

def samples_to_frames(samples, hop_length=512, n_fft=None):
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    samples = np.asanyarray(samples)
    return np.asarray(np.floor((samples - offset) // hop_length), dtype=int)

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


class Translator(multiprocessing.Process):
    def __init__(
        self,
        msg_q: multiprocessing.Queue,
        plot_q: multiprocessing.Queue,
        embedder_params: Dict,
        decision_params: Dict,
        translator_params: Dict,
        stop_event: threading.Event,
        mcast_socket: socket.socket = None,
        load_latest: bool = False,
        save_trace: bool = False,
        state_d: dict = {},
    ):

        super(Translator, self).__init__()
        self.msg_q = msg_q
        self.plot_q = plot_q
        self.state_d = state_d
        state_d["ESPS"] = dict()

        self.stop_event = stop_event

        self.save_trace = save_trace

        self.embedderNetwork = VAENetwork(embedder_params)
        self.translator = SACModel(**translator_params)

        self.mcast_socket = mcast_socket

        self.targets = deque(maxlen=8)

        self.agents = dict()  # {host: agent}
        self.updated = False
        
        self.connections = dict()  # save IP addresses for debugging

        if load_latest:
            self.loadLatest()

        self.session_name = datetime.now().strftime("Session_%Y%m%d_%H%M%S")

        if self.save_trace:
            os.mkdir(os.path.join(TRACE_DIR, self.session_name))

        self.fieldnames = ["time"] + FEATURE_VECTOR_MAP + OUTPUT_VECTOR + ["map_x", "map_y", "active", "highlight"]
        self.observers = dict()

        self.init_networks()
        print("Translator init done")

    def init_networks(self):
        x = np.zeros(len(FEATURE_VECTOR_MAP))
        z = self.embedderNetwork(x)

        a = np.concatenate([z, self.getCenter()])
        self.translator.get_action(a, False)


    def getActiveAgents(self, filter_type: Tuple[ESP] = ()) -> Dict[str, Agent]:
        active = dict()

        now = time()
        for host, agent in self.agents.items():
            if agent.esp_type in (ESP.BLOB, ESP.ESP13):
                agent.active = True
                
            if agent.esp_type in filter_type:
                continue

            if not agent.active:
                continue

            if now > agent.last_ping_time + 30:
                print(f"ESP {agent.id} inactive for 30 seconds, deactivating")
                agent.active = False
                continue

            active[host] = agent
        return active

    @staticmethod
    def makeFeatureVector(features) -> np.ndarray:
        x = [features.get(f, 0) for f in FEATURE_VECTOR_MAP]
        x = np.array(x)

        return x

    def getCenter(self) -> np.ndarray:
        """Returns the coordinate of the center of mass of all the points"""
        positions = []
        for host, agent in self.getActiveAgents(filter_type=(ESP.BLOB, ESP.ESP13)).items():
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
            if agent.esp_type in (ESP.BLOB, ESP.ESP13):
                agent.last_output = {}
                continue

            if not agent.active:
                # print(f"agent {agent.id} inactive, skipping output")
                agent.last_output = {}
                continue

            parameters = dict()
            try:
                y = agent.output_vectors[-1]
            except (IndexError, KeyError):
                agent.last_output = {}
                continue

            for i, name in enumerate(OUTPUT_VECTOR):
                if name in ("temp1", "temp2", "highlight"):
                    parameters[name.lower()] = val
                elif name in ("airon", "airtime"):
                    val = np.clip(y[i], -1, 1)
                    r = OUTPUT_VECTOR_RANGES[name]
                    out = (val / 2.0 + 0.5) * (r[1] - r[0]) + r[0]
                    parameters[name.lower()] = out
                else:
                    val = np.clip(y[i], -1, 1)
                    r = OUTPUT_VECTOR_RANGES[name]
                    out = int((val / 2.0 + 0.5) * (r[1] - r[0]) + r[0])
                    parameters[name.lower()] = out

            data[str(agent.station)] = parameters
            agent.last_output = parameters

        if len(data) > 0:
            self.broadcast(data)


    def broadcast(self, data):
        if self.mcast_socket is None:
            return

        json_data = json.dumps(data)
        sent = self.mcast_socket.sendto(bytes(json_data, 'ascii'), (MCAST_GRP, MCAST_PORT))

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
            last_ping_time=time(),
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


    def updateState(self):
        esp_state = dict()
        for host, agent in self.agents.items():
            id_ = agent.id
            
            esp_state[id_] = {
                "esp_type": agent.esp_type.name,
                "active": agent.active,
                "station": agent.station,
                "port": agent.port,
                "mac": agent.mac,
                # "pos": list(agent.map[-1].astype(float)),
                "last_output": agent.last_output,
                "highlight": agent.highlight,
            }

        self.state_d["ESPS"] = esp_state

    def saveState(self):
        with open("./state.json", "w") as f:
            json.dump(self.state, f, sort_keys=True, indent=4)

    def checkMessages(self):
        while True:
            try:
                (host, port), msg = self.msg_q.get(block=False)
            except Empty:
                break

            if not isinstance(msg, dict):
                continue

            msg_type = msg.get("type", "unknown")

            if msg_type != "update_state":
                self.state_d["last_server_msg"] = msg

            if msg_type == "ping":
                self.handlePingMessage(msg, host, port)
 
            elif msg_type == "sensor":
                self.handleSensorMessage(msg, host, port)
                self.updated = True

            elif msg_type == "raw_sensor":
                self.processRawData(msg, host, port)

            elif msg_type == "whoami":
                if host not in self.agents:
                    print("'whoami' from unregisterwed host ", host)
                    continue
                mac = self.agents[host].mac
                id_ = self.agents[host].id
                station = self.agents[host].station
                data = dict()
                data[mac] = {"type": "whoami", "station": station, "id": id_,}
                self.broadcast(data)

            elif msg_type == "checkpoint":
                self.save()

            elif msg_type == "features":
                self.getFeatures()

            elif msg_type == "output":
                self.output()

            elif msg_type == "agent_positions":
                self.getAgentPositions()

            elif msg_type == "touch_count":
                data = dict()
                data[msg["station"]] = msg
                self.broadcast(data)

                total = self.updateObservers(msg)
                data = dict(ESP13={"type": "observer_count", "observer_count": total})
                self.broadcast(data)

            elif msg_type == "save_state":
                self.saveState()

            elif msg_type == "update_state":
                self.updateState()

            else:
                print(f"Unknown message from {host}, MAC {self.agents.get(host, 'unregistered')}")


    def handlePingMessage(self, msg, host, port):
        if host not in self.agents:
            mac = msg["mac"]
            self.create_agent(mac, host, port)
            return

        self.agents[host].last_ping_time = time()

    def processRawData(self, msg, host, port):
        values = dict(msg)
        for dk, v in msg.items():
            try:
                name = DATA_NAME_MAP[dk]
            except KeyError:
                continue

            if name not in processors:
                continue
            values[dk] = processors[name](np.array(v, dtype=float))

        self.handleSensorMessage(values, host, port)

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

    def updateObservers(self, msg):
        self.observers[msg["station"]] = msg["touch_count"]
        total = sum([x for x in self.observers.values()])
        print(total)
        return total

    def getFeatures(self):
        target = self.getCenter()

        for host, agent in self.getActiveAgents(filter_type=(ESP.BLOB, ESP.ESP13)).items():
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
                # agent.highlight = bool(reward > -0.2)

                self.translator.add(prev_state, action, reward, curr_state, False)
                deterministic = True

            a = self.translator.get_action(curr_state, deterministic)
            a = np.tanh(a * 5)

            agent.output_vectors.append(a)
            agent.map.append(z)

            self.agents[host] = agent

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
        self.updateHighlight()

    def updateHighlight(self, threshold=0.2):
        # first clear the highlight flags (also for inactive agents)
        for host, agent in self.agents.items():
            agent.highlight = False

        active_agents = self.getActiveAgents(filter_type=(ESP.BLOB, ESP.ESP13))
        agents = list(active_agents.keys())
        if len(agents) < 2:
            return

        for i in range(len(agents) - 1):
            host1 = agents[i]
            map1 = active_agents[host1].map[-1]
            for j in range(i+1, len(agents)):
                host2 = agents[j]
                map2 = active_agents[host2].map[-1]

                dist = np.linalg.norm(map2 - map1)
                if dist < threshold:
                    active_agents[i].highlight = True
                    active_agents[j].highlight = True

    def getAgentPositions(self):
        data = dict()
        for host, agent in self.agents.items(): #self.getActiveAgents(filter_type=(ESP.BLOB, ESP.ESP13)).items():
            if agent.esp_type != ESP.HEADSET:
                continue
            data[agent.id] = {"pos": list(agent.map[-1].astype(float)), "active": agent.active}

        if len(data.keys()) == 0:
            return

        data["type"] = "update_points"
        msg = dict(rd=data)
        self.broadcast(msg)


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
        while True:
            self.checkMessages()
            self.updatePlotQueue() 

            if self.stop_event.is_set():
                break

        self.save()

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
        save(data, os.path.join(CHECKPOINT_PATH, fn))

        self.last_checkpoint_time = time()
        print(f"done saving checkpoints ({os.path.join(CHECKPOINT_PATH, fn)})\n")

    def load(self, path: str):
        print("load()")
        try:
            data = load(path)
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
