#! /usr/bin/bash
export PYTHONPATH=`pwd`:$PYTHONPATH
python visualize/vis_rope3d.py  --vis_2d --split training --data_root ./data 