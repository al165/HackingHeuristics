'''
    A simple threaded server with infinite loop, thread-safe message
    passing, and example to shut it down gracefully on 
    KeyboardInterrupt (or any other SIGINT signal)

    Adapted from:
    https://blog.miguelgrinberg.com/post/how-to-kill-a-python-thread
'''

import os
import json
import socket
import datetime
from time import time
from PIL import Image
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event

from typing import Optional, Union

import numpy as np

from config import MCAST_GRP, MCAST_PORT

PICAM_AVAILIABLE = True
try:
    from picamera2 import Picamera2
except ImportError:
    PICAM_AVAILIABLE = False
    print("/!\\ Picamera2 not avaliable")

if PICAM_AVAILIABLE:
    # surpress verbose logging of libcamera and picamera2
    os.environ["LIBCAMERA_LOG_LEVELS"] = "3"
    Picamera2.set_logging(Picamera2.ERROR)


class CameraServer(Thread):
    _wait: float

    def __init__(
        self, 
        stop_event: Event, 
        msg_q: Optional[Queue] = None,
        main_config: Optional[dict] = None,
        lores_config: Optional[dict] = None,
        target_fps: float = 15,
        mcast_socket: socket.socket = None,
        save_root: Union[str,Path] = "./camera_images",
        movement_history: int = 15,
        capture_high: bool = False,
        state_d: dict = {}
    ):
        super(CameraServer, self).__init__()
        global PICAM_AVAILIABLE

        self.stop_event = stop_event
        self.msg_q = msg_q
        self.setFPS(target_fps)
        self.mcast_socket = mcast_socket
        self.save_root = save_root
        self.capture_high = capture_high
        self.state_d = state_d

        os.makedirs(self.save_root, exist_ok=True)

        if main_config is None:
            main_config = {"format": 'XRGB8888', "size": (320, 240)}

        if lores_config is None:
            lores_config = {"format": 'YUV420', "size": (320, 240)}

        if PICAM_AVAILIABLE:
            try:
                self.picam2 = Picamera2()
                self.picam2.configure(
                    self.picam2.create_preview_configuration(
                        main=main_config,
                        lores=lores_config,
                    )
                )
                self.picam2.start()
            except:
                self.picam2 = None
                PICAM_AVAILIABLE = False
        else:
            self.picam2 = None

        self.state_d["camera_server"] = {"picamera2": self.picam2 is not None}

        self.im_main = None
        self.im_lores = None

        self._prev = None
        self.movement = 0
        self._prev_movement = np.zeros(movement_history)

        self._last_capture_time = time()

    def run(self):
        print("CameraServer started")
        while True:
            self.check_messages()

            if self.capture():
                self.measureMovement()

            if self.stop_event.is_set():
                break

        self.close()

    def capture(self) -> bool:
        if self.picam2 is None:
            return False

        now = time()
        if now < self._last_capture_time + self._wait:
            return False

        self._last_capture_time = now
        self._prev = self.im_lores
        (self.im_main, self.im_lores), _ = self.picam2.capture_arrays(["main", "lores"])

        return True

    def measureMovement(self):
        if self.im_lores is None or self._prev is None:
            return

        move = np.square(np.subtract(self.im_lores, self._prev)).mean()
        move /= 90
        move = min(move, 1.0)
        self._prev_movement = np.roll(self._prev_movement, 1)
        self._prev_movement[0] = move
        self.movement = np.mean(self._prev_movement)
        state = self.state_d["camera_server"]
        state["movement"] = self.movement
        self.state_d["camera_server"] = state

    def check_messages(self):
        if self.msg_q is None:
            return

        while True:
            try:
                addr, msg = self.msg_q.get(block=False)
            except Empty:
                break

            msg_type = msg.get("type", "unknown")
            state = self.state_d["camera_server"]
            state["last_camera_msg"] = msg
            self.state_d["camera_server"] = state

            if msg_type == "movement":
                if self.picam2 is None:
                    continue
                data = {"type": "movement", "movement": self.movement}
                self.broadcast(data)

            elif msg_type == "save":
                name = msg.get("name", None)
                self.saveFrame(name)

    def setFPS(self, target_fps: float):
        self.target_fps = target_fps
        self._wait = 1.0/target_fps

    def saveFrame(self, name: Union[Path, str, None]=None):
        if self.im_main is None:
            return

        if name is None:
            name = datetime.datetime.now().strftime("%Y%m%d-%X.png")

        im = Image.fromarray(self.im_main)
        im.save(os.path.join(self.save_root, name))

    def broadcast(self, data):
        if self.mcast_socket is None:
            return

        msg = dict(camera_result=data)
        json_data = json.dumps(msg)
        sent = self.mcast_socket.sendto(bytes(json_data, 'ascii'), (MCAST_GRP, MCAST_PORT))

    def close(self):
        print("\nClosing CameraServer")


def main():
    from signal import signal, SIGINT
    stop_event = Event()

    def signal_handler(signum, frame):
        stop_event.set()
    
    signal(SIGINT, signal_handler)

    msg_q = Queue()

    server = CameraServer(stop_event=stop_event, msg_q=msg_q)
    server.start()
    

if __name__ == "__main__":
    main()
