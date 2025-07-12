#!/bin/bash
# moves the temperature, MDS logs and the save states to mimir_logs
cd /home/asg/

echo "Starting copy"

scp ~/.config/asgard-alignment/instr_states/* asg@192.168.100.1:~/mimir_logs/instr_states
scp ~/Progs/repos/asgard-alignment/data/templogs/* asg@192.168.100.1:~/mimir_logs/temps
scp ~/logs/mds/* asg@192.168.100.1:~/mimir_logs/mds

echo "Copy finished"