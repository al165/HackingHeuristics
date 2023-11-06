import time

import requests
from requests.exceptions import JSONDecodeError

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Container
from textual.widgets import (
    Header, 
    Footer, 
    Pretty, 
    TabbedContent, 
    OptionList,
    TabPane, 
    DataTable, 
    Input, 
    Log, 
    Label,
    Placeholder,
    Static,
)
from textual.reactive import reactive
from textual.widgets.data_table import CellDoesNotExist


URL = "http://localhost:8080/status"


class SendWidget(Static):

    def compose(self) -> ComposeResult:
        yield Log(id="send-hist")
        yield Input(id="send-input")


class Overview(Static):

    data = reactive(dict(), init=False)

    def on_mount(self) -> None:
        self.table = self.query_one(DataTable)
        self.table.add_column("id", key="id")
        self.table.add_column("station", key="station")
        self.table.add_column("type", key="type")
        self.table.add_column("active", key="active")
        self.table.add_column("ip", key="ip")
        self.table.add_column("mac", key="mac")
        self.table.add_column("last_ping", key="ping")

        self.styles.height = '1fr'

        self.set_interval(0.1, self.update_ping)

    def compose(self) -> ComposeResult:
        yield DataTable()

    def update(self, data) -> None:
        self.data = data.get("ESPS", {})

    def watch_data(self) -> None:
        for k, esp in self.data.items():
            active = esp.get("active", None)
            active_str = ''
            if active is None:
                active_str = ''
            elif active:
                active_str = Text("active", "green")
            else:
                active_str = Text("inactive", "red")

            try:
                self.table.update_cell(str(k), "active", active_str)
            except CellDoesNotExist:
                self.table.add_row(
                    k, 
                    esp["station"], 
                    esp["esp_type"], 
                    active_str,
                    esp["ip"],
                    esp["mac"],
                    "-",
                    key=str(k)
                )

        self.table.sort("station", "id")

    def update_ping(self) -> None:
        for k, esp in self.data.items():
            last_ping = time.time() - esp["last_ping_time"]
            string = Text(f"{last_ping:.1f}s")
            if last_ping > 99:
                string = Text(">99s")
            if last_ping > 20:
                string.stylize("bold white on red")
            self.table.update_cell(str(k), "ping", string)
            

class HeadsetWidget(Static):

    data = dict()

    def compose(self) -> ComposeResult:
        yield OptionList(
            "Headset 0", "Headset 1", "Headset 2", "Headset 3", "Headset 4", "Headset 5",
            id="headset-list"
        )
        yield Pretty(self.data, id='headset-pretty')

    def on_option_list_option_highlighted(self, event) -> None:
        out = self.query_one(Pretty)
        index = str(event.option_index)

        out.update(self.data.get(index, {}))

    def update(self, data):
        self.data = data
        options = self.query_one(OptionList)
        for id, esp in data.items():
            if esp["esp_type"] != "HEADSET":
                continue

        for i in range(6):
            if str(i) not in self.data:
                options.disable_option_at_index(i)
            else:
                options.enable_option_at_index(i)


class BlobWidget(Static):

    data = dict()

    def compose(self) -> ComposeResult:
        yield OptionList(
            "Blob 0", "Blob 1", "Blob 2", "Blob 3", "Blob 4", "Blob 5",
            id="blob-list"
        )
        yield Pretty(self.data)

    def on_option_list_option_highlighted(self, event) -> None:
        out = self.query_one(Pretty)
        index = str(event.option_index + 6)

        out.update(self.data.get(index, {}))

    def update(self, data):
        self.data = data
        options = self.query_one(OptionList)
        for id, esp in data.items():
            if esp["esp_type"] != "BLOB":
                continue

        for i in range(6):
            if str(i+6) not in self.data:
                options.disable_option_at_index(i)
            else:
                options.enable_option_at_index(i)


class ESP13Widget(Static):

    def compose(self) -> ComposeResult:
        yield Container(
            Label(Text('status:', justify='right')), Label('offline'),
            Label(Text('last_ping:', justify='right')), Label('--'),
            id='status'
        )
        yield Container(
            Label('valves', id="valves-label"),
            Label('0:0', id='valve-0'),
            Label('1:0', id='valve-1'),
            Label('2:0', id='valve-2'),
            Label('3:0', id='valve-3'),
            Label('4:0', id='valve-4'),
            Label('5:0', id='valve-5'),
            id="valve-view",
        )


class StateViewerApp(App):

    CSS_PATH = "stateViewer.tcss"

    connected = reactive('no server')
    data = reactive(dict(), init=False)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

        with TabbedContent(initial="overview-tab"):
            with TabPane('Overview', id='overview-tab'):
                yield Overview()
            with TabPane('Headsets'):
                yield HeadsetWidget()
            with TabPane('Blobs'):
                yield BlobWidget()
            with TabPane('ESP13'):
                yield ESP13Widget()
            with TabPane('Send', id='id-tab'):
                yield SendWidget()
            with TabPane('Raw Data', id='raw-tab'):
                yield Pretty(self.data, id='pretty_raw')
            with TabPane('Log', id='log-tab'):
                yield Log(id='log')

    def on_mount(self) -> None:
        self.title = "HOTSPOT State Viewer"
        self.set_interval(1, self.get_data)

    def watch_connected(self) -> None:
        self.sub_title = f"status: {self.connected}"

    def watch_data(self) -> None:
        self.query_one('#pretty_raw').update(self.data)
        self.query_one('Overview').update(self.data)
        self.query_one(HeadsetWidget).update(self.data.get("ESPS", {}))

    def add_node(self, tree, data):
        for k, v in data.items():
           if type(v) == dict:
               node = tree.add(f"{k}", expand=False)
               self.add_node(node, v)
           else:
               tree.add_leaf(f"{k}: {v}")

        return tree

    def get_data(self) -> None:
        try:
            r = requests.get(URL, timeout=1)
        except:
            self.connected = 'no server'
            return

        try:
            result = r.json()
        except JSONDecodeError as e:
            self.connected = 'json decode error'
            logger = self.query_one('#log')
            logger.write_line("---")
            logger.write_line(str(r.text))
            logger.write_line(str(e))
            return

        self.connected = 'connected'
        self.data = result


if __name__ == "__main__":
    app = StateViewerApp()
    app.run()