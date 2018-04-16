import random
import re, string
import contextlib
import sys
import logging
import copy
import math
from multiprocessing import Pool
from util import util
from lib import liblinearutil

sample_size = 40

benign_pool = list()
benign_pool_file = "seeds/marvin/training_all.benign"
benign_pool_size = 0

# seed_file = "seeds/marvin/testing_1.seeds"
# seed_file = "seeds/marvin/testing_10.seeds"
seed_file = "seeds/marvin/testing_500.seeds"
model_name = "Marvin/models/model_all_liblinear-L2"

mutation_rate = 0.5
number_unfit = int(sample_size*mutation_rate)
lambd = 0.01

sample_restrict = "D"

perm_cost, static_cost, dynamic_cost = 0.1, 1, 100

header = "500_" + sample_restrict + "_"


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("output/logs/" + name + ".log")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


std_logger = setup_logger("gp/" + header + "master")
feature_logger = setup_logger("gp/" + header + "features")
evasion_logger = setup_logger("gp/" + header + "evasive")


with open("Marvin/features/featurenames", "r") as f:
    feature_names = f.readlines()


def init():
    global benign_pool, benign_pool_size, model
    print("Initializing...")
    random.seed(1)
    model = liblinearutil.load_model(model_name)

    # Load gene pool
    std_logger.debug("Loading gene pool...")
    with open(benign_pool_file, "r") as f:
        population_samples = f.readlines()
    for sample in population_samples:
        benign_pool.append(util.load_marvin(sample))
    benign_pool_size = len(benign_pool)
    std_logger.info("Loaded gene pool")

    print("Initialization complete")


class DummyFile(object):
    def flush(self): pass

    def write(self, x): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


class Experiment:

    def __init__(self):
        self.generation = list()
        self.generation.clear()
        self.min_score = 1.0
        self.best_cumulative_extra = list()
        self.best_cumulative_extra.clear()

    def reset_generation(self, seed):
        # Clear list
        self.generation = list()

        # Set seed score
        with nostdout():
            p_labs, p_acc, p_vals = liblinearutil.predict([seed.label], [seed.features], model, '-b 1')
        seed.score = p_vals[0][1]

        # Initialize BCE
        self.best_cumulative_extra = [seed] * sample_size

    def evaluate_fitness(self, samples, fitness_rate):
        # Sort samples by maliciousness
        samples.sort(key=lambda sample: sample.score)

        # Replace bottom (1-fitness_rate) with fresh samples
        new_sample_index = int(fitness_rate*sample_size)
        for index in range(new_sample_index, len(samples)):
            if samples[index].score < self.best_cumulative_extra[0].score:
                self.best_cumulative_extra.append(samples[index])
                self.best_cumulative_extra.sort(key=lambda sample: sample.score)
                del self.best_cumulative_extra[-1:]
            else:
                samples[index] = self.best_cumulative_extra[0]

        return samples

    def mutate_single(self, sample):
        global benign_pool, feature_names, sample_restrict
        new_sample = copy.deepcopy(sample)
        benign_sample = random.choice(benign_pool)

        num_added_features = 1
        for i in range(num_added_features):
            new_feature = random.choice(list(benign_sample.features.keys()))
            feature_name = feature_names[new_feature]

            if not sample_restrict == "A":
                iterations = 0
                while not feature_name[0] == sample_restrict:
                    new_feature = random.choice(list(benign_sample.features.keys()))
                    feature_name = feature_names[new_feature]
                    iterations += 1
                    if iterations > 10:
                        benign_sample = random.choice(benign_pool)
            else:
                new_feature = random.choice(list(benign_sample.features.keys()))
                feature_name = feature_names[new_feature]

            new_sample.features[new_feature] = 1
            new_sample.added_feat[new_feature] = 1
            feature_logger.info(new_feature)

            # Assign feature cost
            if "PermRequired" in feature_name:
                new_sample.cost += perm_cost
                new_sample.perm_added += 1
            elif feature_name[0] == "S":
                new_sample.cost += static_cost
                new_sample.static_added += 1
            elif feature_name[0] == "D":
                new_sample.cost += dynamic_cost
                new_sample.dynamic_added += 1

            new_sample.fitness = new_sample.score
            if len(new_sample.added_feat) > 0:
                new_sample.fitness += lambd*(new_sample.cost/len(new_sample.added_feat))

        return new_sample

    def mutate_set(self, samples):
        mutation_set = random.sample(range(sample_size), number_unfit)
        for i in mutation_set:
            samples[i] = self.mutate_single(samples[i])
        return samples

    def classify(self, samples):

        # Run model
        generation_input = [sample.features for sample in samples]
        generation_labels = [sample.label for sample in samples]

        with nostdout():
            p_labs, p_acc, p_vals = liblinearutil.predict(generation_labels, generation_input, model, '-b 1')

        for i, val in enumerate(p_vals):
            samples[i].score = val[1]

        samples.sort(key=lambda sample: sample.score)

        generation_best_score = samples[0].score
        if generation_best_score < self.min_score:
            self.min_score = generation_best_score

        std_logger.info("Fitness: " + str(generation_best_score))
        std_logger.debug("----------Sorted samples----------")
        for sample in samples:
            std_logger.debug(sample.stringify())
        std_logger.debug("----------------------------------")

        return samples

    def run_experiment(self, seed, max_gen):
        self.reset_generation(seed)
        current_gen = 0

        for i in range(sample_size):
            self.generation.append(self.mutate_single(seed))
        self.generation = self.classify(self.generation)

        # While evasion performance not good enough or reached max_gen
        while current_gen < max_gen and self.generation[0].score > 0.5:
            self.mutate_set(self.generation)
            self.generation = self.classify(self.generation)
            self.generation = self.evaluate_fitness(self.generation, mutation_rate)

            num_evaded = sum([member.score < 0.5 for member in self.generation])
            generation_string = "Generation complete |" \
                + " Evaded: " + str(num_evaded) + " Mutations: " \
                + str(len(self.generation[0].added_feat.keys()))
            std_logger.info(generation_string)

            current_gen += 1

        if current_gen == max_gen:
            std_logger.warning("Experiment failed - max score: " + str(self.min_score))
            print("Experiment failed - max score: " + str(self.min_score))
        else:
            std_logger.warning("Experiment successful: " + str(current_gen+1))
            std_logger.info("Num features added: " + str(len(self.generation[0].added_feat.keys())))
            print("Final generation",
                  str(current_gen+1), ":",
                  sum([member.score < 0.5 for member in self.generation]))
            summary_string = "Feature types: " + str(self.generation[0].perm_added) \
                             + " " + str(self.generation[0].static_added) \
                             + " " + str(self.generation[0].dynamic_added)
            std_logger.info(summary_string)
            # print(self.generation[0].cost/len(self.generation[0].added_feat))
            for feat in self.generation[0].added_feat.keys():
                evasion_logger.info(feat)
        # print([member.score for member in self.generation])

        return sum([member.score < 0.5 for member in self.generation])


def experiment_set(seed):
    (i, seed_string) = seed
    std_logger.info(["----- RUNNING EXPERIMENT ", i, "-----"])
    # print("Running experiment", str(i), "...")
    exp = Experiment()
    final_fitness = exp.run_experiment(util.load_seed(seed_string), 200)
    # print("Experiment " + str(i) + ": " + str(final_fitness))


if __name__ == "__main__":
    init()
    seed_strings = list()
    with open(seed_file, "r") as f:
        seed_strings = f.readlines()

    with Pool(8) as p:
        p.map(experiment_set, enumerate(seed_strings))
