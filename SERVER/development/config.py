
# IP Address of the ESP controlling the Stimulators
STIM_ADDR = "192.168.0.100"

# i2c Address book:
#    MAC ADDRESS  : (i2c ADDR, CHANNEL, ESP No.)

i2c_ADDRESSES = {
    "C4:4F:33:65:DA:79": (0x1A, 1, 0),  # ESP 0
    "3C:61:05:D1:87:EF": (0x1A, 2, 1),  # ESP 1
    "AA:BB:CC:DD:EE:F2": (0x1B, 1, 2),  # ESP 2
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

