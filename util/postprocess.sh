#!/bin/bash

str=""
for i in `sort -u output/logs/features.log`;
do
    str+=$((i + 1))"p;"
done

sed -n "$str" Marvin/features/featurenames > output/logs/mutations.log
cat output/logs/mutations.log

success=`grep "Success" output/logs/master.log | wc -l`
echo "------------------------------"
echo "Successful:" $success
