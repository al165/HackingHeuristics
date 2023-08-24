# This servers details
HOST = ''
PORT = 8080


# Multicast
MCAST_GRP = '224.3.29.71'
MCAST_PORT = 10000


# ESP and their stations
#         MAC ADDRESS  : (station, id)
STATIONS = {
    "78:21:84:80:A3:E0": (0, 0),   # HEADSET 0
    "EC:62:60:9C:C0:A4": (0, 6),   # BLOB 0
    "3C:61:05:D1:87:EF": (1, 1),   # HEADSET 1
    "40:22:D8:7A:27:D8": (1, 7),   # BLOB 1
    "C0:49:EF:E4:60:BC": (2, 2),   # HEADSET 2
    "EC:62:60:9D:2A:B0": (2, 8),   # BLOB 2
    "94:E6:86:05:13:4C": (3, 3),   # HEADSET 3
    "94:E6:86:03:A3:C4": (3, 9),   # BLOB 3
    "C4:4F:33:65:DA:79": (4, 4),   # HEADSET 4
    "E8:DB:84:C5:C2:B5": (4, 10),  # BLOB 4
    "C4:4F:33:65:DA:80": (5, 5),   # HEADSET 5
    "E8:DB:84:C5:C2:B6": (5, 11),  # BLOB 5
    "FF:FF:FF:FF:FF:FF": (13, 13), # ESP13
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
    "EOG:p_rate",
    "EOG:n_rate",
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
    "temp1",
    "temp2",
    "airon",
    "airtime",
]

OUTPUT_VECTOR_RANGES = {
    "ampl": [3, 6],
    "freq": [1, 100],
    "durn": [0, 2000],
    "idly": [0, 255],
    "temp1": [-2, 2],
    "temp2": [-2, 2],
    "airon": [0, 1],
    "airtime": [0, 10],
}
