import json
import socket
import threading
from queue import Empty

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')

def interp(a, b, x):
    return a + (b - a) * x

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


class RDServer(threading.Thread):

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

    def setFK(f=None, k=None):
        if f is not None:
            self.f = f

        if k is not None:
            self.k = k

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

    def getSamplePoint(self, pos, which='A'):
        i = np.round(np.interp(pos[0], np.linspace(-1, 1, self.N), np.arange(self.N))).astype(int)
        j = np.round(np.interp(pos[1], np.linspace(-1, 1, self.N), np.arange(self.N))).astype(int)

        if which == 'A':
            return self.A[i,j]
        else:
            return self.B[i,j]

    def broadcast(self, data):
        # if self.mcast_socket is None:
        #     return

        msg = dict(rd_samples=data)
        json_data = json.dumps(msg)
        print(json_data)
        # sent = self.mcast_socket.sendto(bytes(msg, 'ascii'), (MCAST_GRP, MCAST_PORT))

    def renderFrame(self, focus='A', save=None):       
        im = self.ax.imshow(self.A, animated=True,vmin=0,cmap='Greys')
        if focus == 'B':
            im.set_array(self.B)
        self.ax.axis('off')
        self.ax.set_title(focus)

        if save is not None:
            plt.savefig(save)

        plt.cla()

    def run(self):
        while True:
            for _ in range(self.steps_per_frame):
                self.update()
            
            if self.save:
                self.renderFrame('B', save=self.save)

            self.check_messages()

            sample_data = self.getSamples()
            self.broadcast(sample_data)

            if self.stop_event.is_set():
                break

    def check_messages(self):
        if self.msg_q is None:
            return

        while True:
            try:
                (host, port), msg = self.msg_q.get(block=False)
            except Empty:
                break

            msg_type = msg.get("type", "unknown")
            if msg_type == "update_point":
                self.addSamplePoint(msg["name"], msg["point"])

            elif msg_type == "remove_point":
                self.removeSamplePoint(msg["name"])

            elif msg_type == "fk":
                self.setFK(msg["f"], msg["k"])

        
def main():
    import signal
    stop_event = threading.Event()

    rd_server = RDServer(stop_event=stop_event)

    rd_server.addSamplePoint("0", [0, 0])
    rd_server.addSamplePoint("1", [-0.6, 0.4])

    def signal_handler(signum, frame):
        print("\nClosing server")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    rd_server.start()


if __name__ == "__main__":
    main()