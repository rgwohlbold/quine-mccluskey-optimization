#!/bin/bash

taskset --cpu-list 9 time ./prime_implicants measure avx2 22
