import subprocess
from mutate_test_opt import mutate

master_file = "seeds/training_all.seeds"
percentages = [0.05, 0.10, 0.25, 0.5, 1.0]
total = 0
with open(master_file, "r") as f:
    for i, l in enumerate(f):
        pass
    total = i + 1

print("Total seeds:", total)

for p in percentages:
    seed_file = "seeds/" + str(p*100) + "%.seeds"
    with open(seed_file, "w+") as f:
        subprocess.run(["head",  "-" + str(int(p*total)), master_file], stdout=f)
    subprocess.run(["util/clean_logs.sh"])
    mutate("Marvin/models/model_all_liblinear-L2", seed_file, "output/evasive/" + str(p*100) + "%.evasive")