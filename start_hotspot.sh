#!/bin/bash

echo "*** Starting HOTSPOT ***"
echo
echo "*** Beginning Server ***"
# cd /home/hotspot/Hacking_Heuristics/HOTSPOT/SERVER/deployment
cd /home/hotspot/HackingHeuristics/HOTSPOT/SERVER/deployment 
python server_new.py --load --trace &
SVR=$!

echo "*** Starting RD ***"
/home/hotspot/openframeworks/apps/myApps/rd_server/bin/rd_server
# /home/arran/Projects/OF/apps/myApps/rd_server/bin/rd_server 

echo "*** Killing Server ***"
kill $SVR
echo "*** HOTSPOT Closed ***"



