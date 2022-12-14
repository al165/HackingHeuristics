# IP Address of the ESP controlling the Stimulators
STIM_ADDR = "192.168.0.100"

# i2c Address book:
#    MAC ADDRESS  : (i2c ADDR, CHANNEL, ESP No.)

i2c_ADDRESSES = {
    "E8:DB:84:C5:C2:B5": (0x1A, 1, 0),  # ESP 0
    "3C:61:05:D1:87:EF": (0x1A, 2, 1),  # ESP 1
    "84:F3:EB:18:3A:25": (0x1B, 1, 2),  # ESP 2
    "AA:BB:CC:DD:EE:F3": (0x1B, 2, 3),  # ESP 3
    "24:6F:28:83:66:50": (0x1C, 1, 4),  # ESP 4
    "AA:BB:CC:DD:EE:F5": (0x1C, 2, 5),  # ESP 5
    "AA:BB:CC:DD:EE:F6": (0x1D, 1, 6),  # ESP 6
    "AA:BB:CC:DD:EE:F7": (0x1D, 2, 7),  # ESP 7
}


# Save directory for model weights and training data
CHECKPOINT_PATH = "../checkpoints"

# Save checkpoint every hour:
CHECKPOINT_INTERVAL = 60 * 60 * 1

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
    "ampl": [3, 12],
    "freq": [1, 100],
    "durn": [0, 2000],
    "idly": [0, 255],
    "temp1": [-1, 1],
    "temp2": [-1, 1],
}
