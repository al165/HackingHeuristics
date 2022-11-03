# HackingHeuristics

Code repository for *Hacking Heuristics* by [Marlot Meyer](https://marlotmeyer.com)

## How to use:

#### Sensor ESP
1. Upload the code `hh_sensors2.ino` in the folder `ESP_CODE/development/` to the ESP32s that handle the sensors.

2. You might want to change the IP address on line 29 to that of the computer's IP address, although this *should* fix itself.

3. Once loaded and powered up, on you phone connect to a wifi network called "HackingHeuristics". You will be directed to a page to set the WiFi password.

4. In the serial monitor, note the MAC address that gets printed.

#### Neurostimduino
1. Upload the code `hh_neurostimduino_2.ino` in the folder `ESP_CODE/development/` to the ESP32 that controls the stimulator.

2. You will also need to set the WiFi as you did with the sensor ESPs.

3. Once started, take a note of the IP address of this ESP.

#### Python Server
1. Edit the IP address of the Neurostimduino on line 48 of `HH_server.py`.

2. Edit the MAC addresses on lines 162-169 in `HH_server.py` to match those of all the sensor ESPs. You can add/delete lines if neccessary.

3. On these lines also change the i2c address numbers to match the stimulators output to the corresponding ESP number.  
:warning: _this assumes that the Neurostimduinos are stacked and the addressing system works as expected - needs to be tested!_ :warning:

4. Open an iTerm window in the folder `SERVER/development/` folder, then run the command:  

        python HH_server.py 8080  
  
    to start the server.
    
    If there is no action and there is an error about "already bound", stop the process by closing the plot window and pressing `ctrl + c`, and try a different port number (between 8080 and 8090).
