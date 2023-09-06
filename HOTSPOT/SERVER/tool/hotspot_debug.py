import json
import socket

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

MCAST_GRP = '224.3.29.71'
MCAST_PORT = 10000

def main():
    multicast_group = (
        MCAST_GRP,
        MCAST_PORT
    )

    mcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mcast_socket.settimeout(0.2)
    mcast_socket.bind(multicast_group)

    history = InMemoryHistory()
    
    history.append_string('{"rd": {"type": "update_points", "0": {"pos": [0.2, -0.5], "active": true}}}')
    history.append_string('{"ESP13": {"type": "observer_count", "observer_count": 6}}')
    history.append_string('{"camera": {"type": "movement"}}')
    history.append_string('{"camera_result": {"movement": 0.4}}')
    history.append_string('{"type": "triggered", "triggered": 0}')
    history.append_string("{\"server\":{\"type\": \"touch_count\", \"touch_count\": 1, \"station\": \"0\"}}")
    history.append_string('{"0":{"airon": 0.75, "airtime": 2}}')
    history.append_string('{"0":{"ampl": 1, "freq": 50, "durn": 500, "idly": 100}}')

    session = PromptSession(history=history, enable_history_search=True)

    print()
    print(" = Welcome to HOTSPOT message center =")
    print("This probably does not work if the server is running")
    print("on the same machine as this one, so either kill it or")
    print("use a different device on the network")
    print(" - CTRL-C, CTRL-D, or 'x' to exit")
    print(" - press Up arrow to scroll through examples/history")
    print(f" - sennding to {MCAST_GRP}:{MCAST_PORT}")

    while True:
        try:
            print()
            msg = session.prompt(">> ")
        except (KeyboardInterrupt, EOFError):
            break

        if msg == 'x':
            break
        elif msg == '':
            continue

        try:
            _ = json.loads(msg)
        except json.JSONDecodeError as e:
            print("!! Json error:")
            print(e)
            print("not sending...")
            continue

        sent = mcast_socket.sendto(bytes(msg, "ascii"), multicast_group)

        print(f"message sent ({sent} bytes)")

    mcast_socket.close()


if __name__ == "__main__":
    main()