import subprocess
from mutate_test_opt import mutate

total = 66891

percentages = [0.01, 0.05, 0.10, 0.25, 0.5]

for p in percentages:
    seed_file = "seeds/" + str(p*100) + "%.seeds"
    with open(seed_file, "w+") as f:
        subprocess.run(["head",  "-" + str(int(p*total)), "seeds/training_all.seeds"], stdout=f)
    mutate(seed_file, "seeds/benign1.seed", "evasive/" + str(p*100) + "%.evasive")