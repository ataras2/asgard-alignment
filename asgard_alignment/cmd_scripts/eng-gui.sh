#! /bin/bash
cd /home/asg/Progs/repos/asgard-alignment
export PYTHONPATH="/home/asg/Progs/repos/asgard-alignment"
xterm -hold -title "Eng Gui Streamlit (installed)" -e streamlit run asgard_alignment/cmd_scripts/m_engineering_GUI.py &