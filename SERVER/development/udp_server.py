import socket

localIP = "0.0.0.0"
localPort = 8084

bufferSize = 1024

UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
UDPServerSocket.bind((localIP, localPort))

print(f'UDP Server listening on {localIP}:{localPort}')

while True:
    msg, addr =  UDPServerSocket.recvfrom(bufferSize)
    if msg == b'HH':
        UDPServerSocket.sendto(b"HH", addr)

    print(f'{addr}: {msg}')

