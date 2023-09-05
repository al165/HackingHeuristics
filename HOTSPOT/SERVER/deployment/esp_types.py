from time import time
from typing import Dict
from collections import deque
from dataclasses import dataclass, field, asdict

from dataclasses_json import dataclass_json, config

import numpy as np

from enum import Enum, auto


class ESP(Enum):
    HEADSET = auto()
    BLOB = auto()
    ESP13 = auto()

class Agent:
    def __init__(
        self,
        ip: str,
        port: int,
        mac: str,
        id: int,
        esp_type: ESP,
        station: int,
    ):
        self.ip = ip
        self.port = port
        self.mac = mac
        self.id = id
        self.esp_type = esp_type
        self.station = station

        self.highlight = False
        self.active = False
        self.last_ping_time = time()
        self.map = deque(maxlen=8)
        self.sensor_data = dict()
        self.sensor_features = dict()
        self.feature_vectors = deque(maxlen=8)
        self.output_vectors = deque(maxlen=8)
        self.last_output = dict()
        self.i2c = 0x1A
        self.touch_count = 0

        self.map.append(np.array([0.0, 0.0]))

    def getSummary(self):
        state = dict(
            ip=self.ip, 
            station=self.station, 
            mac=self.mac, 
            esp_type=self.esp_type.name,
            last_ping_time=self.last_ping_time,
        )

        if self.esp_type == ESP.HEADSET:
            state["highlight"] = self.highlight
            state["active"] = self.active
            state["sensor_data"] = self.sensor_data
            state["sensor_features"] = self.sensor_features
            state["last_output"] = self.last_output

        elif self.esp_type == ESP.BLOB:
            state["i2c"] = f'0x{self.i2c:X}'
            state["touch_count"] = self.touch_count
            state["last_output"] = self.last_output

        return state
