#!/bin/bash

PROCESS_COUNT=$1
DISK_DIR=$2
echo "Run ${PROCESS_COUNT} programs simulteneously..."

for (( i = 1; i <= $PROCESS_COUNT; i++ ))
do
  python3 geom_param_search_5_params_sino.py -d $DISK_DIR -i $i 1>/dev/null 2>"./logs/log-${i}.txt" &
done
