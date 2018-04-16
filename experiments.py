import contextlib
import sys
import os
import random
import copy
import re
import math
import logging
import subprocess
import configparser
from util import util
from lib import liblinearutil
from multiprocessing import current_process

config = configparser.ConfigParser()
section = ''


def cfg_init():
    """
    Initialize the ConfigParser from command line
    """
    if len(sys.argv) < 3:
        print("Error: no config file provided")
        exit(1)
    config.read(sys.argv[1])


def setup_logger(name):
    """
    Set up a single logger instance
    :param name: log file name
    :return: constructed logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("output/logs/" + name + ".log")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


def setup_loggers():
    """
    Set up master, feature, and evasion logger as a group
    :return: 3 constructed loggers
    """
    return setup_logger(config[section]['master_log']), \
           setup_logger(config[section]['feature_log']), \
           setup_logger(config[section]['evasion_log'])


class Experiment:
    """
    Base class for genetic evasion experiments. Contains methods common to genetic process.
    """

    def __init__(self):

        # Initialize and clear generation
        self.generation = list()
        self.generation.clear()

        # Initialize and clear BCE
        self.best_cumulative_extra = list()
        self.best_cumulative_extra.clear()
        self.min_score = 1.0

        # Compute number of unfit samples in each generation
        self.number_unfit = int(config.getint(section, 'sample_size') * config.getfloat(section, 'mutation_rate'))

        self.std_logger, self.feature_logger, self.evasion_logger = setup_loggers()

        random.seed(1)

#TODO: REMOVE
    def set_generation(self, seed):
        """
        Spawn the initial generation list from a seed
        :param seed: Seed for experiment
        """
        self.generation = list()
        seed.score = self.classify([seed])[0]
        self.best_cumulative_extra = [seed] * config.getint(section, 'sample_size')

        # Create initial generation
        for i in range(config.getint(section, 'sample_size')):
            self.generation.append(self.mutate_single(seed))
        self.generation = self.evaluate(self.generation)

    def evaluate_fitness(self, samples):
        """
        Evaluate the fitness of each sample and replace as necessary
        :param samples: Samples to evaluate
        :return: Evaluated samples
        """
        # Sort samples by maliciousness
        samples.sort(key=lambda sample: sample.score)

        for index in range(self.number_unfit, len(samples)):
            # Replace unfit sample with fresh sample from BCE
            if samples[index].score < self.best_cumulative_extra[0].score:
                self.best_cumulative_extra.append(samples[index])
                self.best_cumulative_extra.sort(key=lambda sample: sample.score)
                del self.best_cumulative_extra[-1:]
            # Save sample to BCE
            else:
                samples[index] = self.best_cumulative_extra[0]
        return samples

    def classify(self, samples):
        """
        Query target classifier to obtain classification scores
        :param samples: Samples to classify
        :return: Confidence scores
        """
        raise NotImplementedError("Must be implemented in specific experiment")

    def mutate_single(self, sample):
        """
        Mutate a single sample using experiment-specific mutation procedure
        :param sample: Sample to mutate
        :return: Mutated sample
        """
        raise NotImplementedError("Must be implemented in specific experiment")

    def mutate_set(self, samples):
        """
        Mutate a set of samples
        :param samples: Samples to mutate
        :return: Mutated samples
        """
        mutation_set = random.sample(range(config.getint(section, 'sample_size')), self.number_unfit)
        for i in mutation_set:
            samples[i] = self.mutate_single(samples[i])
        return samples

    def evaluate(self, samples):
        """
        Classify samples and assign scores
        :param samples: Samples to classify
        :return: Scored samples
        """
        # Run target model
        confidences = self.classify(self.generation)

        # Assign scores
        for i, conf in enumerate(confidences):
            samples[i].score = conf

        samples.sort(key=lambda s: s.score/s.cost)

        # Compute best score
        generation_best_score = samples[0].score
        if generation_best_score < self.min_score:
            self.min_score = generation_best_score
        self.std_logger.info("Fitness: " + str(generation_best_score))

        # Log progress
        samples_string = "----------Sorted samples----------\n"
        for sample in samples:
            samples_string += sample.stringify()
        samples_string += "----------------------------------"
        self.std_logger.debug(samples_string)

        return samples

    def run_experiment(self, seed, max_gen):
        """
        Main driver for genetic experiment
        :param seed: Malicious seed for experiment
        :param max_gen: Maximum generations of genetic search
        :return: Best fitness
        """
        self.set_generation(seed)
        current_gen = 0

        # Loop until evading or reaching max_gen
        while current_gen < max_gen and self.generation[0].score > 0.5:  
            self.generation = self.evaluate(self.generation)
            self.generation = self.evaluate_fitness(self.generation)
            self.mutate_set(self.generation)

            num_evaded = sum([member.score < 0.5 for member in self.generation])
            self.std_logger.info("Generation complete |"
                                 + " Evaded: " + str(num_evaded)
                                 + " Mutations: " + str(len(self.generation[0].added_feat.keys())))
            current_gen += 1

        # Experiment over - determine success/failure
        if self.generation[0].score > 0.5:
            self.std_logger.warning("Experiment failed - max score: " + str(self.min_score))
            print("Experiment failed - max score: " + str(self.min_score))
        else:
            self.std_logger.warning("Experiment successful")
            self.std_logger.info("Num features added: " + str(len(self.generation[0].added_feat.keys())))
            print("Completed generation", str(current_gen+1), ":",
                  sum([member.score < 0.5 for member in self.generation]))
            # Record mutations made to evasive sample
            for feat in self.generation[0].added_feat.keys():
                self.evasion_logger.info(feat)

        return sum([member.score < 0.5 for member in self.generation])


class DummyFile(object):
    """
    A dummy file for swallowing Marvin output during classification
    """
    def flush(self): pass

    def write(self, x): pass


@contextlib.contextmanager
def nostdout():
    """
    Stdout swallower for Marvin classification
    """
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


class MarvinExperiment(Experiment):
    """
    Experiment class for Marvin
    """

    is_init = False
    feature_names = []

    @classmethod
    def init(cls):
        """
        Set up config section, benign pool, and feature names
        """
        # Set up config file section
        global section
        section = 'Marvin'
        cls.is_init = True
        cfg_init()

        # Load benign gene pool
        print("Initializing gene pool...")
        with open(config[section]['benign_pool_file'], "r") as f:
            population_samples = f.readlines()
        cls.benign_pool = list()
        for sample in population_samples:
            cls.benign_pool.append(util.load_marvin(sample))

        # Load feature names
        with open(config[section]['featurenames_file'], "r") as f:
            cls.feature_names = f.readlines()

        print("Initialization complete")

    def __init__(self):
        """
        Constructor
        """
        if not self.is_init:
            MarvinExperiment.init()
        super(MarvinExperiment, self).__init__()
        self.model = liblinearutil.load_model(config[section]['model'])
        self.perm_cost, self.static_cost, self.dynamic_cost = 0.1, 1, 100

    def classify(self, samples):
        """
        Query Marvin classifier implemented with LIBLINEAR
        :param samples: Samples to classify
        :return: Confidence scores
        """
        with nostdout():
            labels = [sample.label for sample in samples]
            features = [sample.features for sample in samples]
            p_labs, p_acc, p_vals = liblinearutil.predict(labels, features, self.model, '-b 1')
        return [val[1] for val in p_vals]

    def mutate_single(self, sample):
        """
        Mutate single Marvin sample by adding a randomly-selected feature
        :param sample: Sample to mutate
        :return: mutated_sample
        """
        new_sample = copy.deepcopy(sample)
        benign_sample = random.choice(self.benign_pool)

        num_added_features = 1
        for i in range(num_added_features):
            # Select a feature to add from the benign sample
            new_feature = random.choice(list(benign_sample.features.keys()))
            feature_name = self.feature_names[new_feature]

            # Check for restrictions on allowed mutations
            if not config[section]['sample_restrict'] == "A":
                # Loop while the selected feature is not allowed
                iterations = 0
                while not feature_name[0] == config[section]['sample_restrict']:
                    new_feature = random.choice(list(benign_sample.features.keys()))
                    feature_name = self.feature_names[new_feature]
                    iterations += 1
                    if iterations > 10:
                        benign_sample = random.choice(self.benign_pool)
            else:
                new_feature = random.choice(list(benign_sample.features.keys()))
                feature_name = self.feature_names[new_feature]

            # Add selected feature
            new_sample.features[new_feature] = 1
            new_sample.added_feat[new_feature] = 1
            self.feature_logger.info(new_feature)

            # Assign feature cost
            if "PermRequired" in feature_name:
                new_sample.cost += self.perm_cost
                new_sample.perm_added += 1
            elif feature_name[0] == "S":
                new_sample.cost += self.static_cost
                new_sample.static_added += 1
            elif feature_name[0] == "D":
                new_sample.cost += self.dynamic_cost
                new_sample.dynamic_added += 1

        return new_sample


class CNNExperiment(Experiment):
    """
    Experiment class for CNN model
    """

    file_pattern = re.compile(r"M_.+_(\d+).opseq")
    # Set of logical nop opcodes
    nop_set = ["00", "90", "9b", "a6", "ab",
               "91", "9c", "a7", "ac",
               "92", "9d", "a8", "ad",
               "93", "9e", "a9", "ae"]

    is_init = False
    feature_names = []

    @classmethod
    def init(cls):
        """
        Set up config section
        """
        global section
        section = 'CNN'
        cls.is_init = True
        cfg_init()

    def __init__(self):
        """
        Constructor
        """
        if not self.is_init:
            CNNExperiment.init()
        super(CNNExperiment, self).__init__()

    def classify(self, samples):
        """
        Query CNN classifier implemented in LuaJIT Torch
        :param samples: Samples to classify
        :return: Classifier output
        """
        # Write generation samples to files
        for i, item in enumerate(samples):
            with open(config[section]['eval_dir'] + item.record_id + "_" + str(i) + ".opseq", "w+") as f:
                f.write(item.opcode_sequence())

        # Call Lua script to run model on samples
        os.chdir('deep-android')
        output = subprocess.check_output("th driver.lua -useCUDA -dataDir ./eval/eval_"
                                         + str(current_process()._identity[0]) + " -modelPath ./model.th7", shell=True)
        os.chdir('..')

        # Process confidence scores
        output_list = list(filter(None, output.decode('UTF-8').split("\n")))
        filenames = output_list[:len(self.generation)]

        # Match each sample to its returned confidence score
        generation_indexes = []
        for i, name in enumerate(filenames):
            index_match = self.file_pattern.match(name)
            generation_indexes.append(int(index_match.group(1)))

        # Score format: P(benign), P(malicious)
        scores = output_list[-(1 + len(self.generation)):-1]
        score_split = [[float(score) for score in row.split()] for row in scores]

        # Assign scores
        for i, index in enumerate(generation_indexes):
            self.generation[index].score = score_split[i][1]

        return output

    def mutate_single(self, sample):
        """
        Mutate single CNN sample by adding logical nops
        :param sample: Sample to mutate
        :return: Mutated sample
        """
        new_sample = copy.deepcopy(sample)
        num_features = len(new_sample.features)

        # Randomly choose number of insertion points for logical nops
        num_insertion_pts = int(random.expovariate(1/math.log(num_features)))

        # Insert logical nops in each insertion point
        for i in range(num_insertion_pts):
            # Select insertion point and number of nops
            insertion_line = random.randrange(num_features)
            num_added_nops = int(random.expovariate(1/math.log(num_features)))
            new_sample.opseq_added += num_added_nops

            # Construct nop sequence
            injection = []
            for j in range(num_added_nops):
                injection.append(random.choice(self.nop_set))

            # if insertion_line < num_features:
            insertion_point = random.randrange(len(new_sample.features[insertion_line]))
            opcode_line = new_sample.features[insertion_line]

            # Insert injection into sample opcode list
            new_sample.features[insertion_line] = opcode_line[:insertion_point] \
                + "".join(injection) + opcode_line[insertion_point:]
            self.feature_logger.info((insertion_line, insertion_point))

        return new_sample
