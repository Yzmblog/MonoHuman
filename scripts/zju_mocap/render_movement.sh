#!/usr/bin/env bash

SUBJECT=$1
if [ -z "${SUBJECT}" ]
then
    SUBJECT=387
fi

python run.py \
    --type movement \
    --cfg ./configs/monohuman/zju_mocap/${SUBJECT}/${SUBJECT}.yaml \
    load_net latest
