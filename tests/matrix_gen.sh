#!/bin/bash

testp="gen_tests"
mkdir -p $testp
SEED=10433
percentages=(10 20 40 50 70 90)
dir=
for pct in "${percentages[@]}"; do
    for n in {5..11}; do
        [ -f $testp/rnd-$n-$pct.txt ] && continue
        python tester.py generate $n $pct -s $SEED > $testp/rnd-$n-$pct.txt
        python tester.py verify $testp/rnd-$n-$pct.txt
    done
done

