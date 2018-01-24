import random
import ast, re, string
import logging
import copy
import subprocess
from util.util import *
# from liblinearutil import *

sample_size = 40

gene_pool = list()
gene_pool_file = "features/features_all_labeled_testing"
gene_pool_size = 0

seeds = list()
seed_file = "test1.seeds"

classification_file = "classify.txt"
results_file = "results.txt"

model = "models/model_all_liblinear-L2"

mutation_rate = 0.5
max_feature_num = 496000 #TODO: verify

def init():
    print("Initializing...")
    logging.basicConfig(filename='run.log',level=logging.DEBUG)
    random.seed(1)
    #model = load_model(model_name)

    # Load gene pool
    logging.debug("Loading gene pool...")
    with open(gene_pool_file, "r") as f:
        population_samples = f.readlines()
    for sample in population_samples:
        gene_pool.append(load_record(sample))
    gene_pool_size = len(gene_pool)
    logging.info("Loaded gene pool")

    # Select intial working set
    # working_set_indices = random.sample(gene_pool_size, sample_size)
    # for index in working_set_indices:
    #     working_set.append(gene_pool[int(index)])
    print("Initialization complete")


class Experiment:

    working_set = list()
    max_score = 0.0

    def reset_working_set(self, seed):
        for i in range(sample_size):
            sample = self.mutate_single(seed)
            self.working_set.append(sample)
            logging.debug(sample.stringify())
        logging.info("Set working set for seed")

    def evaluate_fitness(self, samples, fitness_rate):
        # Sort samples by maliciousness
        sorted_samples = sorted(samples, key=lambda sample: sample.score, reverse=True)

        # Replace bottom (1-fitness_rate) with fresh samples
        # new_sample_index = fitness_rate*sample_size
        # while new_sample_index < len(samples):
        #     sorted_samples[new_sample_index] = gene_pool[random.randrange(gene_pool_size)]
        #     new_sample_index += 1

        return sorted_samples

    def mutate_single(self, sample):
        new_sample = copy.deepcopy(sample)
        new_feature_index = random.randrange(2, max_feature_num)
        new_sample.features[new_feature_index] = 1
        return new_sample

    def mutate_set(self, samples):
        # mutation_set = random.sample(range(sample_size), int(sample_size*mutation_rate))
        for i in range(int(sample_size*mutation_rate)):
            samples[-i] = self.mutate_single(samples[i])

        return samples

    def classify(self, samples):
        global max_score
        with open(classification_file, "w+") as f:
            for sample in samples:
                f.write(sample.sparse_arff())

        # Run model
        # Python interface: p_label, p_acc, p_val = predict(y, x, model, '-b 1')
        result = subprocess.run(["./lib/predict", "-b", "1", classification_file, model, results_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        logging.info("Prediction output: " + output)
        print(output)
        failure_percentage = 100 - int(re.findall(r'(\d+)\%', output)[0][:-1])

        with open(results_file, "r") as f:
            for sample in samples:
                sample.score = 1.0 - float(f.readline().split(" ")[1])
                if sample.score > self.max_score:
                    self.max_score = sample.score
        logging.info("Fitness: " + str(failure_percentage))
        logging.debug("Sorted samples")
        logging.debug("----------") 
        for sample in samples:
            logging.debug(sample.stringify())
        logging.debug("----------")

        return failure_percentage

    def run_experiment(self, seed, max_gen):
        self.reset_working_set(seed)
        current_gen = 0
        batch_fitness = self.classify(self.working_set)

        #while evasion performance not good enough or reached max_gen
        while current_gen < max_gen and batch_fitness < 100: 
            working_set = self.evaluate_fitness(self.working_set, mutation_rate)
            self.mutate_set(working_set)
            batch_fitness = self.classify(working_set)
            current_gen += 1

        if current_gen == max_gen:
            logging.warning("Experiment failed - max score: " + str(self.max_score))
            print("Experiment failed - max score: " + str(self.max_score))
        else:
            logging.warning("Experiment successful")

        return batch_fitness


if __name__ == "__main__":
    init()
    seed_strings = list()
    with open(seed_file, "r") as f:
        seed_strings = f.readlines()
    for i, seed_string in enumerate(seed_strings):
        logging.info(["----- RUNNING EXPERIMENT ", i, "-----"])
        print("Running experiment " +  str(i) + "...")
        exp = Experiment()
        final_fitness = exp.run_experiment(load_record(seed_string), 100)
        print("Experiment " + str(i) + ":" + str(final_fitness))
