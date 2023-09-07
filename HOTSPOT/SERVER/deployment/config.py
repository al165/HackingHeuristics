from esp_types import ESP

# This servers details
HOST = ''
PORT = 8080
TCP_PORT = 8081

# Multicast
MCAST_GRP = '224.3.29.71'
MCAST_PORT = 10000

########################
# ESP and their stations
#         MAC ADDRESS  : (station, id, type)
STATIONS = {
    "EC:62:60:9C:C0:A4": (0, 0, ESP.HEADSET),   # HEADSET 0
    "24:D7:EB:15:19:34": (0, 6, ESP.BLOB),      # BLOB 0
    "40:22:D8:7A:27:D8": (1, 1, ESP.HEADSET),   # HEADSET 1
    "24:D7:EB:15:18:00": (1, 7, ESP.BLOB),      # BLOB 1
    "A0:B7:65:DD:FB:D4": (2, 2, ESP.HEADSET),   # HEADSET 2
    "40:22:D8:7A:33:10": (2, 8, ESP.BLOB),      # BLOB 2
    "78:21:84:80:A3:E0": (3, 3, ESP.HEADSET),   # HEADSET 3
    "A4:CF:12:97:1D:2C": (3, 9, ESP.BLOB),      # BLOB 3
    "C0:49:EF:E4:60:BC": (4, 4, ESP.HEADSET),   # HEADSET 4
    "24:6F:28:A5:08:DC": (4, 10, ESP.BLOB),     # BLOB 4
    "FC:F5:C4:0F:BE:6C": (5, 5, ESP.HEADSET),   # HEADSET 5
    "24:6F:28:83:66:50": (5, 11, ESP.BLOB),     # BLOB 5
    "40:22:D8:EA:A3:40": (13, 13, ESP.ESP13),   # ESP13
    "E8:DB:84:C5:C2:B5": (13, 13, ESP.ESP13),   # TESTING ESP
    "C4:4F:33:65:DA:79": (13, 13, ESP.ESP13),   # TESTING ESP
    "30:AE:A4:F3:48:94": (1, 7, ESP.BLOB),   # TESTING ESP
    "3C:71:BF:58:FA:A8": (1, 1, ESP.HEADSET),   # TESTING ESP
}
########################

BLOB_NS_ADDRS = {
    0: 0x1A,
    1: 0x1A,
    2: 0x1A,
    3: 0x1C,
    4: 0x1A,
    5: 0x1B,
}


# Save directory to store data traces
TRACE_DIR = "../sessions"

# data to store in trace
#TRACE_FIELDS = ["time", "EEG:mean"]

# Save directory for model weights and training data
CHECKPOINT_PATH = "../checkpoints"

# Save checkpoint every 4 hours:
#CHECKPOINT_INTERVAL = 4
CHECKPOINT_INTERVAL = 60 * 60 * 4

# Update feature vectors every 4 seconds:
UPDATE_TIME = 4

# Data information
DATA_KEYS = ["data0", "data1", "data2", "data3"]

DATA_NAME_MAP = {
    "data0": "EEG",
    "data1": "EOG",
    "data2": "heart",
    "data3": "GSR",
}

DATA_UNITS = {
    "data0": "frames",
    "data1": "samples",
    "data2": "samples",
    "data3": "samples",
}

# Inputs
FEATURE_VECTOR_MAP = [
    "EEG:mean",
    "EEG:delta",
    "EOG:all_rate",
    # "EOG:p_rate",
    # "EOG:n_rate",
    "GSR:mean",
    "GSR:delta",
    "heart:std",
]

# Outputs
OUTPUT_VECTOR = [
    "ampl",
    "freq",
    "durn",
    "idly",
    # "temp1",
    # "temp2",
    "airon",
    "airtime",
]

OUTPUT_VECTOR_RANGES = {
    "ampl": [1, 3],
    "freq": [1, 100],
    "durn": [0, 2000],
    "idly": [0, 255],
    "temp1": [-1, 1],
    "temp2": [-1, 1],
    "airon": [0, 1.0],
    "airtime": [1, 5],
}

RD_COLOR_MAP = "plasma"
