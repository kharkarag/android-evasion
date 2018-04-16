from multiprocessing import Pool
from experiments import *

seed_dir = "seeds/opseq_seeds/test/"

def marvin_exp(seed):
    (i, seed_string) = seed
    # print("Running experiment", str(i), "...")
    exp = MarvinExperiment()
    final_fitness = exp.run_experiment(util.load_seed(seed_string), 200)
    # print("Experiment " + str(i) + ": " + str(final_fitness))

def cnn_exp(seed):
    (i, seed_file) = seed
    with open(seed_dir + seed_file, "r") as f:
        opcodes = f.read().splitlines()

    exp = Experiment()
    final_fitness = exp.run_experiment(util.load_opseq(opcodes, 1, seed_file), 300)
    subprocess.run("rm deep-android/eval/*", shell=True)
    # print("Experiment " + str(i) + ": " + str(final_fitness))

if __name__ == "__main__":
    seed_strings = list()
    with open(sys.argv[2], "r") as f:
        seed_strings = f.readlines()

    # with Pool(8) as p:
    #     p.map(experiment_set, enumerate(seed_strings))

    for seed in enumerate(seed_strings):
        marvin_exp(seed)
