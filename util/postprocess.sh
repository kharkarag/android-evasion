#!/bin/bash

str=""
for i in `sort -u output/logs/$1evasive.log`;
do
    str+=$((i + 1))"p;"
done

sed -n "$str" Marvin/features/featurenames > output/logs/$1mutations.log
cat output/logs/$1mutations.log

success=`grep "successful" output/logs/$1master.log | wc -l`
echo "------------------------------"
echo "Successful:" $success
