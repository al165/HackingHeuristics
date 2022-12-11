from queue import Empty
from typing import List, Dict, Tuple

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from config import (
    DATA_KEYS,
    DATA_NAME_MAP,
    FEATURE_VECTOR_MAP,
    OUTPUT_VECTOR,
    OUTPUT_VECTOR_RANGES,
)


matplotlib.rcParams["toolbar"] = "None"
plt.style.use("dark_background")
matplotlib.rcParams["axes.edgecolor"] = "#AAA"
matplotlib.rcParams["xtick.color"] = "#AAA"
matplotlib.rcParams["ytick.color"] = "#AAA"
matplotlib.rcParams["text.color"] = "#AAA"

CMAP = matplotlib.cm.get_cmap("rainbow")


AX_LIMS = {
    "EEG": (-1.1, 1.1),
    "EOG": (-1.1, 1.1),
    "GSR": (-1.1, 1.1),
    "heart": (-1.1, 1.1),
}


class MinPlotter:
    def __init__(self, plot_q):
        self.plot_q = plot_q
        self.fig = plt.figure()

        self.map_ax = self.fig.add_subplot()

        self.map_ax.set_ylim(-1.1, 1.1)
        self.map_ax.set_xlim(-1.1, 1.1)
        self.map_ax.set_xticks([])
        self.map_ax.set_yticks([])

        self.agents = dict()
        self.colors = dict()

        self.lines = dict()

    def animate(self, num: int) -> List[Line2D]:
        all_lines: List[Line2D] = list()
        updated = False

        while True:
            try:
                agent = self.plot_q.get(block=False)
                # self.plot_data.update(plot_data)
                id = agent.id
                if id not in self.agents:
                    self.agents[id] = agent
                    c = CMAP(len(self.colors) / 8)
                    self.colors[id] = c
                updated = True
            except Empty:
                break

        if not updated:
            return all_lines

        for id, agent in self.agents.items():
            m = np.stack(agent.map)
            xs = m[:, 0]
            ys = m[:, 1]

            if id not in self.lines["map"]:
                line = Line2D(xs, ys, color=self.colors[id])
                self.lines["map"][id] = line
                self.map_ax.add_line(line)
            else:
                self.lines["map"][id].set_data(xs, ys)

            all_lines.append(self.lines["map"][id])

        # draw lines connecting all highlighted participants
        highlighted = []
        for id, agent in self.agents.items():
            if agent.highlight:
                highlighted.append(id)

        for i in range(len(highlighted) - 1):
            for j in range(i + 1, len(highlighted)):
                id1 = highlighted[i]
                id2 = highlighted[j]
                start = self.agents[id1].map[-1]
                end = self.agents[id2].map[-1]

                line = Line2D(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    lw=2,
                    ls="--",
                    c="gold",
                )

                all_lines.append(line)

        return all_lines


class FullPlotter:
    def __init__(self, plot_q):
        self.plot_q = plot_q

        self.fig = plt.figure()
        self.gs = GridSpec(4, 3, width_ratios=[1, 2, 2])
        self.gs.tight_layout(self.fig)

        self.sensor_axes = [
            self.fig.add_subplot(self.gs[0, 0]),
            self.fig.add_subplot(self.gs[1, 0]),
            self.fig.add_subplot(self.gs[2, 0]),
            self.fig.add_subplot(self.gs[3, 0]),
        ]

        for dk, ax in zip(DATA_KEYS, self.sensor_axes):
            name = DATA_NAME_MAP[dk]
            ax.set_ylim(*AX_LIMS[name])
            ax.set_xlim(-4, 0)
            ax.set_title(name)

        self.map_ax = self.fig.add_subplot(self.gs[:2, 1:])
        self.embed_loss_ax = self.fig.add_subplot(self.gs[2, 1])
        self.trans_loss_ax = self.fig.add_subplot(self.gs[2, 2])
        self.feat_vector_ax = self.fig.add_subplot(self.gs[3, 1])
        self.out_vector_ax = self.fig.add_subplot(self.gs[3, 2])

        self.map_ax.set_title("map")
        self.map_ax.set_ylim(-1.1, 1.1)
        self.map_ax.set_xlim(-1.1, 1.1)
        self.map_ax.set_xticks([])
        self.map_ax.set_yticks([])

        self.embed_loss_ax.set_title("embedding loss")
        self.embed_loss_ax.set_xlim(0, 16)
        self.embed_loss_ax.set_xticks([])
        self.trans_loss_ax.set_title("translator loss")
        self.trans_loss_ax.set_xlim(0, 16)
        self.trans_loss_ax.set_xticks([])

        self.feat_vector_ax.set_title("feature vector")
        self.feat_vector_ax.set_xticks([])
        self.feat_vector_ax.set_yticks(np.arange(len(FEATURE_VECTOR_MAP)))
        self.feat_vector_ax.set_yticklabels(FEATURE_VECTOR_MAP)

        self.out_vector_ax.set_title("output vector")
        self.out_vector_ax.set_xticks([])
        self.out_vector_ax.set_yticks(np.arange(len(OUTPUT_VECTOR)))
        self.out_vector_ax.set_yticklabels(OUTPUT_VECTOR)

        self.plot_data = {
            "sensors": dict(),
            "map": dict(),
            "feature_vectors": dict(),
            "output_vectors": dict(),
            "embedder_losses": [],
            "translator_losses": {
                "a_losses": [],
                "q_losses": [],
            },
        }

        self.lines = self.plot_data.copy()

        em_l = Line2D([], [], color="white")
        self.lines["embedder_losses"] = em_l
        self.embed_loss_ax.add_line(em_l)

        a_l = Line2D([], [], color="orange")
        self.lines["translator_losses"]["a_losses"] = a_l
        self.trans_loss_ax.add_line(a_l)
        q_l = Line2D([], [], color="aqua")
        self.lines["translator_losses"]["q_losses"] = q_l
        self.trans_loss_ax.add_line(q_l)

        feature_mat = np.zeros((1, len(FEATURE_VECTOR_MAP)))
        f_mat_show = self.feat_vector_ax.imshow(feature_mat.T, vmin=0, vmax=100)
        # self.feat_vector_ax.set_xlim(0, 8)
        self.lines["feature_vectors"] = f_mat_show

        output_mat = np.zeros((1, len(OUTPUT_VECTOR_RANGES)))
        o_mat_show = self.out_vector_ax.imshow(output_mat.T, vmin=-1, vmax=1)
        # self.out_vector_ax.set_xlim(0, 8)
        self.lines["output_vectors"] = o_mat_show

        self.colors = dict()

    def animate(self, num: int) -> List[Line2D]:
        """
        plot_data:
          - 'sensors'
              - <host>
                  - 'EEG' = np.ndarray
                  - 'EOG' = np.ndarray
                  - 'GSR' = np.ndarray
                  - 'heart' = np.ndarray
          - 'map'
              - <host> = list
          - 'feature_vectors'
              - <host> = list
          - 'output_vectors'
              - <host> = list
          - 'training_losses'
              - 'embedding' = list
              - 'intrepretter' = list
              - 'translator'
                - 'a_loss' = list
                - 'q_loss' = list
        """

        all_lines: List[Line2D] = []
        updated = False

        while True:
            try:
                plot_data = self.plot_q.get(block=False)
                self.plot_data.update(plot_data)
                updated = True
            except Empty:
                break

        if not updated:
            return all_lines

        # --------------- SENSORS ----------------
        for dk, ax in zip(DATA_KEYS, self.sensor_axes):
            # if "sensors" not in plot_data:
            #    break

            name = DATA_NAME_MAP[dk]
            for host in self.plot_data["sensors"].keys():
                buffers = self.plot_data["sensors"][host][name]

                t = t_f
                if DATA_UNITS[dk] == "samples":
                    t = t_sr

                if host not in self.colors:
                    c = CMAP(len(self.colors) / 8)
                    self.colors[host] = c

                if host not in self.lines["sensors"]:
                    self.lines["sensors"][host] = dict()

                if name not in self.lines["sensors"][host]:
                    line = Line2D(
                        t,
                        buffers,
                        linewidth=1,
                        alpha=0.5,
                        color=self.colors[host],
                    )
                    self.lines["sensors"][host][name] = line
                    ax.add_line(line)
                else:
                    self.lines["sensors"][host][name].set_data(t, buffers)

                all_lines.append(self.lines["sensors"][host][name])

        # --------------- MAP ----------------
        for host in self.plot_data["map"].keys():
            # if "map" not in plot_data:
            #    break

            m = np.stack(self.plot_data["map"][host])
            xs = m[:, 0]
            ys = m[:, 1]

            if host not in self.lines["map"]:
                line = Line2D(xs, ys, color=self.colors[host])
                self.lines["map"][host] = line
                self.map_ax.add_line(line)
            else:
                self.lines["map"][host].set_data(xs, ys)

            all_lines.append(self.lines["map"][host])

        # --------------- FEATURES ----------------
        # if "feature_vectors" in plot_data:
        fvs = self.plot_data["feature_vectors"]
        feature_mat = np.zeros((8, len(FEATURE_VECTOR_MAP)))
        for i, host in enumerate(self.colors.keys()):
            if host in fvs and len(fvs[host]) > 0:
                fv = fvs[host][-1]
                feature_mat[i] = fv

        if len(feature_mat):
            self.feat_vector_ax.matshow(feature_mat.T, vmin=-1, vmax=20)

        # if "output_vectors" in plot_data:
        ovs = self.plot_data["output_vectors"]
        output_mat = np.zeros((8, len(OUTPUT_VECTOR)))
        for i, host in enumerate(self.colors.keys()):
            if host in fvs and len(ovs[host]) > 0:
                ov = ovs[host][-1]
                output_mat[i] = ov

        if len(output_mat):
            # self.lines["output_vectors"].set_data(output_mat.T)
            # self.out_vector_ax.set_xlim(0, 8)
            self.out_vector_ax.matshow(output_mat.T, vmin=-1, vmax=1)

        # --------------- LOSSES ----------------
        # if "embedder_losses" in plot_data:
        em_losses = self.plot_data["embedder_losses"]
        if em_losses:
            self.lines["embedder_losses"].set_data(
                np.arange(len(em_losses)),
                em_losses,
            )
            all_lines.append(self.lines["embedder_losses"])
            self.embed_loss_ax.set_ylim(0, np.max(em_losses) * 1.1)

        # if "translator_losses" in plot_data:
        a_losses = self.plot_data["translator_losses"]["a_losses"]
        q_losses = self.plot_data["translator_losses"]["q_losses"]

        if a_losses:
            self.lines["translator_losses"]["a_losses"].set_data(
                np.arange(len(a_losses)), a_losses
            )
            all_lines.append(self.lines["translator_losses"]["a_losses"])
            self.lines["translator_losses"]["q_losses"].set_data(
                np.arange(len(q_losses)), q_losses
            )
            all_lines.append(self.lines["translator_losses"]["q_losses"])
            upper = max(np.max(a_losses), np.max(q_losses))
            lower = min(np.min(a_losses), np.min(q_losses))
            self.trans_loss_ax.set_ylim(
                lower - abs(0.1 * lower),
                upper + abs(0.1 * upper),
            )

        self.fig.canvas.draw()

        return all_lines
