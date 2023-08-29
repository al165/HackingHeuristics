import json
import socket
import threading
import multiprocessing
from PIL import Image
from queue import Empty

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from config import MCAST_GRP, MCAST_PORT, RD_COLOR_MAP

plt.switch_backend('agg')

def interp(a, b, x):
    return a + (b - a) * x

def map_range(x, x1, x2, y1, y2, clip=False):
    p = (x - x1) / (x2 - x1)
    if clip:
        p = min(max(p, 0.0), 1.0)
    y = y1 + (y2 - y1)*p
    return y

def xy_to_kf(x, y):
    ''' 
    Implemented map function from Karl Sims RD Tool 
    https://karlsims.com/rdtool.html
    '''
    y1 = y * 0.5 + 0.5
    f = interp(0.002, 0.12, y1)
    s = np.sqrt(f) * 0.5 - f
    x1 = x * interp(1, (y - 0.32) * (y - 0.32), 0.6) * 0.5 + 0.5
    k0 = interp(-0.003, 0.0115, x1)
    k1 = interp(-0.0048, -0.0031, x1)
    k = s + interp(k0, k1, y1)

    return k, f

def apply_laplacian(mat):
    neigh_mat = -4*mat.copy()
    neighbors = [ 
                    ( 1.0,  (-1, 0) ),
                    ( 1.0,  ( 0,-1) ),
                    ( 1.0,  ( 0, 1) ),
                    ( 1.0,  ( 1, 0) ),
                ]

    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (1,0))

    return neigh_mat

def get_initial_A_and_B(N, random_influence = 0.1):
    """get the initial chemical concentrations"""

    # get initial homogeneous concentrations
    A = (1-random_influence) * np.ones((N,N))
    B = np.zeros((N,N))

    # put some noise on there
    A += random_influence * np.random.random((N,N))
    B += random_influence * np.random.random((N,N))

    # get center and radius for initial disturbance
    N2, r = N//2, 50

    # apply initial disturbance
    A[N2-r:N2+r, N2-r:N2+r] = 0.50
    B[N2-r:N2+r, N2-r:N2+r] = 0.25

    return A, B


class RDServer(multiprocessing.Process):

    def __init__(
        self, 
        N=200, 
        steps_per_frame=30, 
        msg_q=None, 
        stop_event=None, 
        mcast_socket=None,
        save=None,
    ):

        super(RDServer, self).__init__()

        self.msg_q = msg_q
        self.mcast_socket=mcast_socket
        self.save = save

        self.N = N
        self.A = np.ones((N, N))
        self.B = np.zeros((N, N))

        for _ in range(16):
            self.addRandom()

        self.delta_t = 1.0

        self.DA = 0.16
        self.DB = 0.08

        self.f = 0.060
        self.k = 0.062

        self.sample_points = dict()

        self.steps_per_frame = steps_per_frame

        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        self.stop_event = stop_event

    def update(self):
        diff_A = self.DA * apply_laplacian(self.A)
        diff_B = self.DB * apply_laplacian(self.B)
        
        reaction = self.A*self.B**2
        diff_A -= reaction
        diff_B += reaction

        diff_A += self.f * (1-self.A)
        diff_B -= (self.k+self.f) * self.B

        self.A += diff_A * self.delta_t
        self.B += diff_B * self.delta_t

    def setFK(self, f=None, k=None):
        if f is not None:
            self.f = f

        if k is not None:
            self.k = k

    def setXY(self, x, y):
        k, f = xy_to_kf(x, y)
        self.setFK(f, k)

    def addRandom(self):
        r = 5

        N1 = np.random.randint(self.N)
        N2 = np.random.randint(self.N)

        self.A[N1-r:N1+r, N2-r:N2+r] = 0.50
        self.B[N1-r:N1+r, N2-r:N2+r] = 0.25

    def addSamplePoint(self, name, pos):
        self.sample_points[name] = np.array(pos)

    def removeSamplePoint(self, name):
        del self.sample_points[name]

    def getSamples(self, which='A'):
        data = dict()
        for i, sp in self.sample_points.items():
            data[i] = self.getSamplePoint(sp, which)
        return data

    def posToIndex(self, pos):
        # i = np.round(np.interp(pos[0], np.linspace(-1, 1, self.N), np.arange(self.N))).astype(int)
        # j = np.round(np.interp(pos[1], np.linspace(-1, 1, self.N), np.arange(self.N))).astype(int)
        i = int(map_range(pos[0], -1, 1, 0, N, True))
        j = int(map_range(pos[1], -1, 1, 0, N, True))

        return i, j

    def getSamplePoint(self, pos, which='A'):
        i, j = self.posToIndex(pos)

        if which == 'A':
            return self.A[i,j]
        else:
            return self.B[i,j]

    def broadcast(self, data):
        if self.mcast_socket is None:
            return

        msg = dict(server=data)
        json_data = json.dumps(msg)
        sent = self.mcast_socket.sendto(bytes(json_data, 'ascii'), (MCAST_GRP, MCAST_PORT))

    def renderFrame(self, focus='A', save=None):
        data = None
        if focus == 'A':
            data = np.clip(np.array(self.A), 0, 1)
        else:
            data = np.clip(np.array(self.B), 0, 1)

        cmap = mpl.colormaps[RD_COLOR_MAP]

        #im = Image.fromarray(np.uint8(cmap(data)*255))
        #im.save(save)

        im = np.uint8(cmap(data)*255)
        cv2.imshow("rd_server", im)



    def run(self):
        cv2.startWindowThread()
        # cv2.namedWindow("rd_server", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("rd_server", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            for _ in range(self.steps_per_frame):
                self.update()
            
            if self.save:
                self.renderFrame('B', save=self.save)

                if cv2.waitKey(0):
                    self.save = None
                    cv2.destroyAllWindows()

            self.check_messages()

            if self.stop_event.is_set():
                break
        cv2.destroyAllWindows()

    def check_messages(self):
        if self.msg_q is None:
            return

        while True:
            try:
                (host, port), msg = self.msg_q.get(block=False)
            except Empty:
                break

            msg_type = msg.get("type", "unknown")

            if msg_type == "update_points":
                for name, pos in msg.items():
                    if name == "type":
                        continue

                    if not msg["active"]:
                        self.removeSamplePoint(name)
                    else:
                        self.addSamplePoint(name, msg["pos"])

            elif msg_type == "remove_point":
                self.removeSamplePoint(msg["name"])

            elif msg_type == "fk":
                self.setFK(msg["f"], msg["k"])

            elif msg_type == "xy":
                self.setXY(msg["x"], msg["y"])

            elif msg_type == "get_samples":
                if len(self.sample_points) == 0:
                    continue
                data = self.getSamples()
                self.broadcast(data)

            elif msg_type == "add_random":
                self.addRandom()

            elif msg_type == "movement":
                movement = msg["movement"]
                k = map_range(movement, 2, 70, 0.0463, 0.0646, True)
                f = map_range(movement, 2, 70, 0.0137, 0.0501, True)

                self.setFK(f, k)
                print(f"updating FK: {movement:.3f} -> {self.f:.3f}, {self.k:.3f}")


def main():

    print(map_range(2, 2, 70, 1.0, 0.1, True), 1.0)
    print(map_range(35, 2, 70, 1.0, 0.1, True), 0.5)
    print(map_range(60, 2, 70, 1.0, 0.1, True),2)
    print(map_range(0.0, 2, 70, 1.0, 0.1, True),2)
    print(map_range(0.0, 2, 70, 1.0, 0.1, True),2)


    import signal
    stop_event = threading.Event()

    rd_server = RDServer(stop_event=stop_event, save="rd.png")

    rd_server.addSamplePoint("0", [0, 0])
    rd_server.addSamplePoint("1", [-0.6, 0.4])

    def signal_handler(signum, frame):
        print("\nClosing server")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    rd_server.start()


if __name__ == "__main__":
    main()
