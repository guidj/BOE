#!/usr/bin/env bash

set -xe

function run(){

    DIR=$(dirname $0)
    PY=$DIR/../py
    python -m cProfile -o syndicato.cprof $PY/syndicato/experimenting/context_free_epsilon_greedy_bernoulli.py --report-path=$HOME/egreedy.html --logs-path=$HOME/egreedy.csv
}

run "$@"
