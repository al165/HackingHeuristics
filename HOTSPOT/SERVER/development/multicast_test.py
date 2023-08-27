import socket
import struct
import sys
import time
import json


data = {
    "0": {
        "type": "sensor",
        "touchCount": 1,
    }
}

message = bytes(json.dumps(data), "ascii")
print(message)

# message = b'{"C4:4F:33:65:DA:79": {"data": 123}}'
multicast_group = (
    '224.3.29.71',
    10000
)

server_address = (
    '',
    10000
)


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(0.2)
sock.bind(server_address)

group = socket.inet_aton(multicast_group[0])
# mreq = struct.pack('4s4s', group, socket.inet_aton('192.168.2.9'))
mreq = struct.pack('4sL', group, socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

ttl = struct.pack('b', 1)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)

last_lighthouse = 0

try:
    print(f"sending {message}")
    sent = sock.sendto(message, multicast_group)

    print("waiting to receive...")
    while True:
        if time.time() > last_lighthouse + 30:
            sent = sock.sendto(b"{'lighthouse': {'port': 8000}}", multicast_group)
            last_lighthouse = time.time()
        try:
            data, server = sock.recvfrom(1024)
        except socket.timeout:
            continue
        else:
            print(f"received {data.decode('utf-8')} from {server}")
except KeyboardInterrupt:
    sock.close()
finally:
    print("closing socket")
    sock.close()
