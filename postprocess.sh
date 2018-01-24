#!/bin/bash

str=""
for i in `sort -u logs/features.log`;
do
    str+=$((i + 1))"p;"
done

sed -n "$str" Marvin/features/featurenames > assisted_mutation/mutations.log
cat assisted_mutation/mutations.log

success=`grep "Success" logs/master.log | wc -l`
echo "------------------------------"
echo "Successful:" $success
