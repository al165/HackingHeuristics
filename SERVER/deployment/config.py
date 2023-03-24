# IP Address of the ESP controlling the Stimulators
STIM_ADDR = "192.168.178.68"

# i2c Address book:
#    MAC ADDRESS  : (i2c ADDR, CHANNEL, ESP No.)

i2c_ADDRESSES = {
    "78:21:84:80:A3:E0": (0x1B, 1, 0),  # ESP 0
    "EC:62:60:9C:C0:A4": (0x1D, 2, 1),  # ESP 1
    "24:D7:EB:14:F8:70": (0x1B, 2, 2),  # ESP 2
    "40:22:D8:7A:27:D8": (0x1C, 1, 3),  # ESP 3
    "C0:49:EF:E4:60:BC": (0x1D, 1, 4),  # ESP 4
    "EC:62:60:9D:2A:B0": (0x1A, 2, 5),  # ESP 5
    "94:E6:86:05:13:4C": (0x1A, 1, 6),  # ESP 6
    #"24:D7:EB:14:F1:08": (0x1A, 1, 6),  # ESP 6
    "94:E6:86:03:A3:C4": (0x1C, 2, 7),  # ESP 7
    "C4:4F:33:65:DA:79": (0x1A, 1, 0),  # DEBUG ESP 0
    "E8:DB:84:C5:C2:B5": (0x1A, 2, 1),  # DEBUG ESP 1
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
]

OUTPUT_VECTOR_RANGES = {
    "ampl": [3, 6],
    "freq": [1, 100],
    "durn": [0, 2000],
    "idly": [0, 255],
    "temp1": [-2, 2],
    "temp2": [-2, 2],
}
