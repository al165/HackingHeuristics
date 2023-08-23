- Existing function for turning led on and off in neurostim library ( can be turned on using command ‘ON’ and ‘OFF’) 
- I changed the ArduinoLedPin number from 2 to pin 15 in the .h file to control the valve with command ‘ON’ or ‘OFF’ 
- In the same file I also added the onboard led (pin 2) to be triggered by the command   
- Each headset + handset now has it’s own stimulation ESP, which also controls the valve inflating the shape belonging to that set.  
- Technically the hands and the head receive the same output (because the hands should feel what the headset person is feeling) but maybe the amplitude needs to be different  
- Two hands share 1 channel, but they will only work if one hand of one person is used, if both hands are used at the same time the circuit changes

- Channel 1 = headset 
- Channel 2 = hands  
- So that means there are 6 headset esp’s & 6 corresponding stim/valve esp’s.  
    - These valves should not stay open for longer than 15 seconds (1 MINUTE OFF TIME) 
- There is also a 13th esp, which controls all 6 valves for the air tube installation. Here, each headset is allocated one output pin. Output pins are:  26, 16, 17, 18, 19,  23 

- These valves can stay open longer (maybe max one minute and then a break) 

- Esp 13 also has a little screen for debugging purposes - i.e print which valves are open or something