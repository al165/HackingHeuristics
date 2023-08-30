from time import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict

from dataclasses_json import dataclass_json, config

import numpy as np

from enum import Enum, auto


class ESP(Enum):
    HEADSET = auto()
    BLOB = auto()
    ESP13 = auto()


@dataclass_json
@dataclass
class Agent:
    ip: str
    port: int
    mac: str
    id: int
    esp_type: ESP
    map: deque = field(
        default_factory=lambda: deque(maxlen=8)
    )  # , metadata=config(encoder=list))
    active: bool = False
    highlight: bool = False
    station: int = -1
    sensor_data: Dict = field(default_factory=dict)
    feature_vectors: deque = field(
        default=deque(maxlen=8), metadata=config(encoder=list)
    )
    output_vectors: deque = field(
        default=deque(maxlen=8), metadata=config(encoder=list)
    )
    last_output: dict = {},
    last_ping_time: float = field(default=time())

    def __post_init__(self):
        self.map.append(np.array([0.0, 0.0]))