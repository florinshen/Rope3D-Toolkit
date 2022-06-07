#! /usr/bin/bash
export PYTHONPATH=`pwd`:$PYTHONPATH
python visualize/vis_rope3d.py  --vis_2d --split validation --data_root ./data --scale 1.0
