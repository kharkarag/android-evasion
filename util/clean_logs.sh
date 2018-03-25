#!/bin/bash

rm output/logs/$1master.log
rm output/logs/$1features.log
rm output/logs/$1evasive.log
rm output/logs/$1mutations.log

if [ $# -gt 1 ] && [ $2 == '-cnn' ];
then
    rm deep-android/eval/*;

