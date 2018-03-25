import re
import sys
import numpy as np

if len(sys.argv) > 1:
    filename = "output/logs/" + sys.argv[1] + "master.log"
else:
    filename = "output/logs/master.log"

with open(filename, "r") as f:
    log = f.readlines()

mutation_re = re.compile("Mutations: +(\d+)")
feature_re = re.compile("Feature types: *(\d+) (\d+) (\d+)")

mutations = []
perm, static, dynamic = [], [], []

for line in log:
    mutation_match = mutation_re.search(line)
    if not mutation_match == None:
        mutations.append(int(mutation_match.group(1)))
    feature_match = feature_re.search(line)
    if not feature_match == None:
        perm.append(int(feature_match.group(1)))
        static.append(int(feature_match.group(2)))
        dynamic.append(int(feature_match.group(3)))


print(len(mutations))
print("Mutations", np.mean(mutations))
print("Perm", np.mean(perm))
print("Static", np.mean(static))
print("Dynamic", np.mean(dynamic))