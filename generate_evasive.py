import sys
import contextlib
import subprocess
import copy
import logging
import random
from util import *
from lib import liblinearutil

# start_feat = int(sys.argv[1])
# end_feat = int(sys.argv[2])

model = liblinearutil.load_model("Marvin/models/model_all_liblinear-L2")

std_logger = logging.getLogger("standard_logger")
std_logger.setLevel(logging.DEBUG)
std_fh = logging.FileHandler("logs/generator/master.log")
std_fh.setLevel(logging.DEBUG)
std_logger.addHandler(std_fh)

feature_logger = logging.getLogger("feature_logger")
feature_logger.setLevel(logging.DEBUG)
feature_fh = logging.FileHandler("logs/generator/features.log")
feature_fh.setLevel(logging.DEBUG)
feature_logger.addHandler(feature_fh)

class DummyFile(object):
    def flush(self): pass
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

def mutate(malicious_file, benign_file, evasive_output):
    with open(benign_file, "r") as f:
        benign_samples = f.readlines()
    benign_sample_size = len(benign_samples)

    print("------------------------------")
    # print("Total features: " + str(len(benign.features.keys())))
    # print("------------------------------")

    with open(malicious_file, "r") as f:
        malicious_samples = f.readlines()

    evasive_file = open(evasive_output, "w+")

    malicious_sample_size = len(malicious_samples)

    for sample_num, malicious_sample in enumerate(malicious_samples):

        malicious = load_record(malicious_sample)
        malicious_orig = copy.deepcopy(malicious)

        benign = load_record(benign_samples[random.randrange(benign_sample_size)])
        benign_list = list(benign.features.keys())

        best_probs = [1.0, 0.0]
        best_feat = -1

        std_logger.debug("------------------------------")
        # print("------------------------------")
        std_logger.info(malicious_orig.stringify())
        for i in range(15):

            mutation_list = []
            mutation_labels = []

            for feat in benign_list:
                malicious.features[feat] = 1

                mutation_list.append(malicious.features)
                mutation_labels.append(malicious.label)

                malicious = copy.deepcopy(malicious_orig)

            with nostdout():
                p_labs, p_acc, p_vals = liblinearutil.predict(mutation_labels, mutation_list, model, '-b 1')

            best_feat_index, best_probs = min(enumerate(p_vals), key=lambda vals: vals[1][0])
            best_feat = benign_list[best_feat_index]

            # print("Feat: " + str(best_feat) + ", prob: " + str(best_probs))
            std_logger.info(str(best_feat) + ": " + str(best_probs))
            feature_logger.info(str(best_feat))
            malicious_orig.features[best_feat] = 1
            malicious = copy.deepcopy(malicious_orig)

            if best_probs[0] < 0.5:
                std_logger.warning("Success | Final: " + str(best_probs[0]) + " | Mutations: " + str(i+1))
                evasive_file.write(malicious.sparse_arff())
                break
                # print("Success - final prob: " + str(best_probs[0]))

        if best_probs[0] > 0.5:
            std_logger.warning("Failure | Final prob: " + str(best_probs[0]))

        if sample_num % int(malicious_sample_size/20) == 0:
            print("Progress: " + str(round(sample_num/malicious_sample_size*100)) + "%")

    evasive_file.close()
    print("------------------------------")
    subprocess.run("./postprocess.sh")
    print("------------------------------")

if __name__ == '__main__':
    random.seed(10001)
    mutate(sys.argv[1], sys.argv[2], sys.argv[3])
