import re
import numpy as np

with open("output/logs/master.log", "r") as f:
    log = f.readlines()

r = re.compile("Mutations: +(\d+)")

mutations = []

for line in log:
    match = r.search(line)
    if not match == None:
        mutations.append(int(match.group(1)))

print(len(mutations))
print("Mean", np.mean(mutations))