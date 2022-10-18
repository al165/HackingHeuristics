# HackingHeuristics

Code repository for *Hacking Heuristics* by [Marlot Meyer](https://marlotmeyer.com)

## How to use:

#### Sensor ESP
Upload the code 'hh_sensors.ino' in the folder `ESP_CODE/development/` to the ESP32s that handle the sensors.

(You might need to change the IP address on line 11 to the computer's IP address, although this *should* fix itself.)

Once loaded and powered up, on you phone connect to a wifi network called "HackingHeuristics". You will be directed to a page to set the WiFi password.

#### Neurostimduino
Upload the code 'hh_neurostimduino.ino' in the folder `ESP_CODE/development/` to the ESP32 that controls the stimulator.

You will also need to set the WiFi as you did with the sensor ESPs.

Once started, take a note of the IP address of this ESP.

#### Python Server
Edit the IP address of the Neurostimduino on line 36 of `HH_server.py`.

Open an iTerm window in the folder `SERVER/development/` folder, then run the command:
```
python HH_server.py 8080
```
to start the server.

If there is no action and there is an error about "already bound", stop the process by closing the plot window and pressing `ctrl+c`.
