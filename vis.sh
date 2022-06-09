#! /usr/bin/bash
export PYTHONPATH=`pwd`:$PYTHONPATH
python visualize/vis_rope3d.py --split validation --data_root ./data --scale 1.0
