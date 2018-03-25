import random
import re, string
import contextlib
import sys
import os
import logging
import subprocess
import copy
import math
import numpy as np
from multiprocessing import Pool
from util import util
from lib import liblinearutil

sample_size = 40

random.seed(1)

seed_dir = "seeds/opseq_seeds/test/"
eval_dir = "deep-android/eval/"

mutation_rate = 0.5
number_unfit = int(sample_size*mutation_rate)

header = ""

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("output/logs/" + name + ".log")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger

std_logger = setup_logger("gp_cnn/" + header + "master")
feature_logger = setup_logger("gp_cnn/" + header + "features")
evasion_logger = setup_logger("gp_cnn/" + header+ "evasive")

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


    def evaluate(self):
        for i, item in enumerate(self.generation):
            with open(eval_dir + item.sample_id + "_" + str(i) + ".opseq", "w+") as f:
                f.write(item.opcode_sequence())

        # Call lua script to run model on samples
        os.chdir('deep-android')
        output = subprocess.check_output("th driver.lua -useCUDA -dataDir ./eval -modelPath ./model.th7", shell=True)
        os.chdir('..')

        output_list = list(filter(None, output.decode('UTF-8').split("\n")))
        scores = output_list[1:-1]
        score_split = [[float(score) for score in row.split()] for row in scores]
        print(score_split)
        # Each score is P(benign), P(malicious)

        for i, score in enumerate(score_split):
            # self.generation[i].score = 1 - score[0]/score[1]
            self.generation[i].score = score[1]

        return output

    def reset_generation(self, seed):
        self.generation = list()
        self.generation.append(seed)

        seed.score = self.evaluate()
        self.best_cumulative_extra = [seed] * sample_size

    def evaluate_fitness(self, samples, fitness_rate):
        # Sort samples by maliciousness
        samples.sort(key = lambda sample: sample.score)

        # Replace bottom (1-fitness_rate) with fresh samples
        new_sample_index = int(fitness_rate*sample_size)
        for index in range(new_sample_index, len(samples)):
            if samples[index].score < self.best_cumulative_extra[0].score:
                self.best_cumulative_extra.append(samples[index])
                self.best_cumulative_extra.sort(key = lambda sample: sample.score)
                del self.best_cumulative_extra[-1:]
            else:
                samples[index] = self.best_cumulative_extra[0]
        #     sorted_samples[new_sample_index] = gene_pool[random.randrange(gene_pool_size)]

        return samples

    def mutate_single(self, sample):
        new_sample = copy.deepcopy(sample)
        num_features = len(new_sample.features)

        num_added_nops = int(random.expovariate(1/math.log10(num_features)))
        insertion_point = random.randrange(num_features + 1)

        new_sample.features[insertion_point:insertion_point] = [0] * num_added_nops
        feature_logger.info(insertion_point)

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

        self.evaluate()

        samples.sort(key = lambda sample: sample.score)

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
            self.generation = self.classify(self.generation)
            self.generation = self.evaluate_fitness(self.generation, mutation_rate)
            self.mutate_set(self.generation)

            num_evaded = sum([member.score < 0.5 for member in self.generation])
            std_logger.info("Generation complete | Evaded: " + str(num_evaded) + " Mutations: " + str(len(self.generation[0].added_feat_indices)))

            # if (current_gen+1) % 5 == 0:
            #     print([member.score for member in self.generation])
                # print("Completed generation", str(current_gen+1), ":", sum([member.score < 0.5 for member in self.generation]))
            print("Generation", str(current_gen+1), self.generation[0].score)
            current_gen += 1

        if current_gen == max_gen:
            std_logger.warning("Experiment failed - max score: " + str(self.min_score))
            print("Experiment failed - max score: " + str(self.min_score))
        else:
            std_logger.warning("Experiment successful")
            std_logger.info("Num features added: " + str(len(self.generation[0].added_feat_indices)))
            print("Completed generation", str(current_gen+1), ":", sum([member.score < 0.5 for member in self.generation]))
            for feat in self.generation[0].added_feat_indices:
                evasion_logger.info(feat)
        # print([member.score for member in self.generation])

        return sum([member.score < 0.5 for member in self.generation])


def experiment_set(seed):
    (i, seed_file) = seed
    with open(seed_dir + seed_file, "r") as f:
        opcodes = f.read().splitlines()

    std_logger.info(["----- RUNNING EXPERIMENT ", i, "-----"])
    exp = Experiment()
    final_fitness = exp.run_experiment(util.load_opseq(opcodes, 1, seed_file), 300)
    # print("Experiment " + str(i) + ": " + str(final_fitness))

if __name__ == "__main__":
    seed_files = []
    for seed in os.listdir(seed_dir):
        seed_files.append(seed)

    # with Pool(8) as p:
    #     p.map(experiment_set, enumerate(seed_files))
    for seed in seed_files:
        experiment_set((0, seed))

    
