#!/bin/python

import json
import struct
import socket
import argparse
import threading
import multiprocessing
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler

from translator import Translator, processors

from config import (
    HOST,
    PORT,
    MCAST_GRP,
    MCAST_PORT,
    UPDATE_TIME,
    OUTPUT_VECTOR,
    DATA_NAME_MAP,
    CHECKPOINT_INTERVAL,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", 
    "--port",
    metavar="port",
    type=int, 
    default=8080, 
    required=False,
    help="port to start HTTP listening server on",
)
parser.add_argument(
    "--load",
    action="store_true",
    default=False,
    required=False,
    help="load latest model. Default False.",
)
parser.add_argument(
    "-t", 
    "--trace",
    default=False, 
    required=False, 
    action="store_true", 
    help="save .csv files tracing agent's state",
)
parser.add_argument(
    "--host", 
    metavar="host",
    type=str,
    default="",
    required=False, 
    help="host IP to start HTTP listening server on",
)


embedder_params = {
    "feature_size": 8,
    "hidden_size": [16, 8],
    "latent_size": 2,
}

decision_params = {
    "sizes": [2, 8, 12, 8],
    "batchnorm": True,
}

translator_params = {
    "state_dim": embedder_params["latent_size"] * 2,
    "action_dim": len(OUTPUT_VECTOR),
    "gamma": 0.99,
    "hid_shape": (12, 12),
    "a_lr": 0.0001,
    "c_lr": 0.0001,
    "batch_size": 4,
    "alpha": 0.12,
    "adaptive_alpha": False,
}


def processBuffer(data):
    try:
        values = json.loads(data)
        return values
    except Exception as e:
        print(e)
        return {"data0": [], "data1": [], "data2": [], "data3": [], "active": False, "type": "sensor"}


def makeHTTPServer(msg_q):
    class HH_HttpServer(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.msg_q = msg_q
            super(HH_HttpServer, self).__init__(*args, **kwargs)

        def _set_response(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                json_data = post_data.decode('ascii')
            except:
                print("error decoding data")
                self._set_response()

            addr = self.client_address
            values = processBuffer(json_data)
            if "server" not in values:
                self._set_response()
                return

            values = values["server"]

            for dk, v in values.items():
                try:
                    name = DATA_NAME_MAP[dk]
                except KeyError:
                    continue

                if name not in processors:
                    continue

                values[dk] = processors[name](np.array(v, dtype=float))

            msg_q.put((addr, values))
            self._set_response()

        def log_request(self, code='-', size='-'): 
            return
    
    return HH_HttpServer


def lighthouse(mcast_socket):
    sent = mcast_socket.sendto(
        bytes("{\"all\": {\"type\": \"lighthouse\", \"port\": " + str(PORT) + "}}", 'ascii'), 
        (MCAST_GRP, MCAST_PORT)
    )
        
def multicast_listener(mcast_socket, stop_event, mailboxes):
    while True:
        if stop_event.is_set():
            break

        try:
            data, addr = mcast_socket.recvfrom(1024)
        except socket.timeout:
            continue
        else:
            try:
                json_data = json.loads(data.decode('ascii'))
                print(json_data)
            except json.decoder.JSONDecodeError:
                print("error parsing", data.decode('ascii'))

            for recipiant, queue in mailboxes.items():
                if recipiant in json_data:
                    queue.put((addr, json_data[recipiant]))



def getInterfaceIP(interface="wlan0"):
    from subprocess import check_output

    output = check_output(["ip", "-j", "addr", "show", interface])
    d = json.loads(output)

    try:
        host = d[0]['addr_info'][0]['local']
    except (IndexError, KeyError):
        print(f"** Cannot establish interface {interface} IP address. Defaulting to localhost")
        host = '127.0.0.1'

    return host


def main():
    import signal
    from rd_server import RDServer

    global HOST, PORT

    args = parser.parse_args()
    if args.port:
        PORT = args.port

    if args.host:
        HOST = args.host
    elif HOST == '':
        HOST = getInterfaceIP()

    multicast_group = (
        MCAST_GRP,
        MCAST_PORT
    )

    mcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mcast_socket.settimeout(0.2)
    mcast_socket.bind(multicast_group)

    print()
    print("="*50)
    print(f"  Multicast group:   {MCAST_GRP}:{MCAST_PORT}")
    print("="*50)
    print()

    group = socket.inet_aton(multicast_group[0])
    mreq = struct.pack('4sL', group, socket.INADDR_ANY)
    mcast_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    ttl = struct.pack('b', 1)
    mcast_socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
    
    stop_event = threading.Event()
    mailboxes = dict()
    server_q = multiprocessing.Queue()
    mailboxes["server"] = server_q


    def signal_handler(signum, frame):
        print("\nClosing server")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    def add_message(msg):
        server_q.put(((HOST, PORT), msg))

    scheduler = BackgroundScheduler()
    scheduler.add_job(lighthouse, 'interval', args=(mcast_socket,), seconds=10)
    scheduler.add_job(add_message, 'interval', args=({"type": "checkpoint"},), seconds=CHECKPOINT_INTERVAL)
    scheduler.add_job(add_message, 'interval', args=({"type": "features"},), seconds=UPDATE_TIME)
    scheduler.add_job(add_message, 'interval', args=({"type": "output"},), seconds=UPDATE_TIME)

    translator = Translator(
        server_q,
        None,
        embedder_params,
        decision_params,
        translator_params,
        mcast_socket=mcast_socket,
        load_latest=args.load,
        save_trace=args.trace,
        stop_event=stop_event,
    )

    rd_q = multiprocessing.Queue()
    mailboxes["rd"] = rd_q
    rd_server = RDServer(
        stop_event=stop_event, 
        mcast_socket=mcast_socket, 
        msg_q=rd_q,
        save="rd.png",
    )

    handler_class = makeHTTPServer(server_q)
    httpd = HTTPServer((HOST, PORT), handler_class)
    http_thread = threading.Thread(target=httpd.serve_forever)
    http_thread.daemon = True

    print()
    print("="*50)
    print(f"  HTTP Server:       {HOST}:{PORT}")
    print("="*50)
    print()

    rd_server.start()
    translator.start()
    scheduler.start()
    http_thread.start()

    print()
    print("READY")
    print()
    print("Received messages:")
    print("-"*50)

    multicast_listener(mcast_socket, stop_event, mailboxes)

    print("\nCLOSING")

    httpd.shutdown()
    scheduler.shutdown()
    translator.join()

    print("\nDONE")


if __name__ == "__main__":
    main()