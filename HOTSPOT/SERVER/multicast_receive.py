import socket
import struct

MCAST_GRP = '224.3.29.71'
MCAST_PORT = 10000
IS_ALL_GROUPS = False

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP, fileno=None)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

if IS_ALL_GROUPS:
    # on this port, receives ALL multicast groups
    sock.bind(('', MCAST_PORT))
else:
    # on this port, listen ONLY to MCAST_GRP
    sock.bind((MCAST_GRP, MCAST_PORT))

mreq = struct.pack("=4s4s", socket.inet_aton(MCAST_GRP), socket.inet_aton("192.168.2.9"))
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

while True:
    print(sock.recv(1024))




# import socket
# import multicast_expert



# print(multicast_expert.get_default_gateway_iface_ip_v4())
# print("listening...")
# while True:
#     with multicast_expert.McastRxSocket(
#         socket.AF_INET, mcast_ips=['224.3.29.71'], 
#         port=10000, 
#         iface_ip=multicast_expert.get_default_gateway_iface_ip_v4()
#     ) as mcast_rx_sock:

#         data, src_address = mcast_rx_sock.recvfrom()
#         print(data.decode("utf-8"), src_address)