import contextlib
import sys
import os
import random
import copy
import math
import subprocess

from util import util
from lib import liblinearutil

# TODO: logs


class Experiment:

    def __init__(self):
        self.generation = list()
        self.generation.clear()
        self.min_score = 1.0
        self.best_cumulative_extra = list()
        self.best_cumulative_extra.clear()

        self.sample_size = 40
        self.mutation_rate = 0.5
        self.number_unfit = int(self.sample_size * self.mutation_rate)

    def reset_generation(self, seed):
        self.generation = list()
        seed.score = self.classify()[0]
        self.best_cumulative_extra = [seed] * self.sample_size

    def evaluate_fitness(self, samples, fitness_rate):
        # Sort samples by maliciousness
        samples.sort(key=lambda sample: sample.score)

        # Replace bottom (1-fitness_rate) with fresh samples
        new_sample_index = int(fitness_rate * self.sample_size)
        for index in range(new_sample_index, len(samples)):
            if samples[index].score < self.best_cumulative_extra[0].score:
                self.best_cumulative_extra.append(samples[index])
                self.best_cumulative_extra.sort(key = lambda sample: sample.score)
                del self.best_cumulative_extra[-1:]
            else:
                samples[index] = self.best_cumulative_extra[0]
        return samples

    def classify(self, samples):
        raise NotImplementedError("Must be implemented in specific experiment")

    def mutate_single(self, sample):
        raise NotImplementedError("Must be implemented in specific experiment")

    def mutate_set(self, samples):
        mutation_set = random.sample(range(self.sample_size), self.number_unfit)
        for i in mutation_set:
            samples[i] = self.mutate_single(samples[i])
        return samples

    def evaluate(self, samples):
        # Run model
        confidences = self.classify(self.generation)

        for i, conf in enumerate(confidences):
            samples[i].score = conf[1]

        samples.sort(key=lambda sample: sample.score/sample.cost)

        generation_best_score = samples[0].score
        if generation_best_score < self.min_score:
            self.min_score = generation_best_score

        std_logger.info("Fitness: " + str(generation_best_score))

        samples_string = "----------Sorted samples----------\n"
        for sample in samples:
            samples_string += sample.stringify()
        samples_string += "----------------------------------"
        std_logger.debug(samples_string)

        return samples

    def run_experiment(self, seed, max_gen):
        self.reset_generation(seed)
        current_gen = 0

        for i in range(self.sample_size):
            self.generation.append(self.mutate_single(seed))
        self.generation = self.evaluate(self.generation)

        # While evasion performance not good enough or reached max_gen
        while current_gen < max_gen and self.generation[0].score > 0.5:  
            self.generation = self.evaluate(self.generation)
            self.generation = self.evaluate_fitness(self.generation, mutation_rate)
            self.mutate_set(self.generation)

            num_evaded = sum([member.score < 0.5 for member in self.generation])
            std_logger.info("Generation complete | Evaded: " + str(num_evaded) + " Mutations: " + str(len(self.generation[0].added_feat.keys())))
            current_gen += 1

        if current_gen == max_gen:
            std_logger.warning("Experiment failed - max score: " + str(self.min_score))
            print("Experiment failed - max score: " + str(self.min_score))
        else:
            std_logger.warning("Experiment successful")
            std_logger.info("Num features added: " + str(len(self.generation[0].added_feat.keys())))
            print("Completed generation", str(current_gen+1), ":", sum([member.score < 0.5 for member in self.generation]))
            for feat in self.generation[0].added_feat.keys():
                evasion_logger.info(feat)
        # print([member.score for member in self.generation])

        return sum([member.score < 0.5 for member in self.generation])


class DummyFile(object):
    def flush(self): pass

    def write(self, x): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


class MarvinExperiment(Experiment):

    def __init__(self, model_name):
        super(MarvinExperiment, self).__init__()
        self.model = liblinearutil.load_model(model_name)

    def classify(self, samples):
        with nostdout():
            labels = [sample.label for sample in samples]
            features = [sample.features for sample in samples]
            p_labs, p_acc, p_vals = liblinearutil.predict(labels, features, self.model, '-b 1')
        return [val[1] for val in p_vals]

    def mutate_single(self, sample):
        global benign_pool
        new_sample = copy.deepcopy(sample)

        benign_sample = random.choice(benign_pool)

        # num_added_features = random.randrange(len(benign_sample.features.keys()))
        num_added_features = int(random.expovariate(1/math.log(len(benign_sample.features.keys()))))
        for i in range(num_added_features):
            new_feature = random.choice(list(benign_sample.features.keys()))
            new_sample.features[new_feature] = 1
            new_sample.added_feat[new_feature] = 1
            feature_logger.info(new_feature)

            # Assign feature cost
            feature_name = feature_names[new_feature]
            if "PermRequired" in feature_name:
                cost += 0.1
            elif feature_name[0] == "S":
                cost += 1.0
            elif feature_name[0] == "D":
                cost += 10.0

        return new_sample

class CNNExperiment(Experiment):

    def __init__(self):
        super(CNNExperiment, self).__init__()

    def classify(self, samples):
        for i, item in enumerate(samples):
            with open(eval_dir + item.record_id + "_" + str(i) + ".opseq", "w+") as f:
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

        return score_split

    def mutate_single(self, sample):
        new_sample = copy.deepcopy(sample)
        num_features = len(new_sample.features)

        num_added_nops = int(random.expovariate(1/math.log(num_features)))
        insertion_point = random.randrange(num_features + 1)

        new_sample.features[insertion_point:insertion_point] = [0] * num_added_nops
        feature_logger.info(insertion_point)

        return new_sample
