import numpy as np

with open("output/logs/gp_features.log", "r") as f:
    features = f.readlines()

with open("Marvin/features/featurenames", "r") as f:
    featurenames = [p[:-1] for p in f.readlines()]

features = list(map(int, features))

unique = np.unique(features, return_counts=True)
print(unique)

unique_features = set(features)
print("Unique: " + str(len(unique_features)))

names = []
for feat in features:
    names.append(featurenames[feat])

print(len(names))

for h in unique[0][np.argsort(unique[1])][-20:]:
    print(featurenames[h])