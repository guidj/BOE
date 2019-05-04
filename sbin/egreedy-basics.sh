#!/usr/bin/env bash

set -xe

function run(){

    DIR=$(dirname $0)
    PARENT_DIR=$DIR/..
    PY=$PARENT_DIR/py
    export PYTHONPATH=$PYTHONPATH:$PY

    python -m syndicato.experimenting \
        --config-file=$PARENT_DIR/experiments/egreedy-basics.yaml \
        --job-dir=$HOME/exp/bandits/ \
        --nproc=4
}

run "$@"
