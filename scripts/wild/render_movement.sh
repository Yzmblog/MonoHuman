#!/usr/bin/env bash

SUBJECT=$1
if [ -z "${SUBJECT}" ]
then
    SUBJECT=monocular
fi

python run.py \
    --type movement \
    --cfg ./configs/monohuman/wild/${SUBJECT}/wild.yaml \
    load_net latest
