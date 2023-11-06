#!/bin/bash

echo "*** Starting HOTSPOT ***"
echo
echo "*** Beginning Server ***"
echo
# cd /home/hotspot/Hacking_Heuristics/HOTSPOT/SERVER/deployment
cd /home/hotspot/HackingHeuristics/HOTSPOT/SERVER/deployment 
/usr/bin/python3 server_new.py --load --trace &
SVR=$!
echo "server PID: $SVR"

echo "*** Starting RD ***"
echo
/home/hotspot/openframeworks/apps/myApps/rd_server/bin/rd_server
# /home/arran/Projects/OF/apps/myApps/rd_server/bin/rd_server 

echo "*** Killing Server ***"
kill $SVR
echo
echo "*** HOTSPOT Closed ***"