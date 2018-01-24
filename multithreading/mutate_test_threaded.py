import sys
import contextlib
import subprocess
import copy
import logging
import threading
import subprocess
from util import *
from lib import liblinearutil

# start_feat = int(sys.argv[1])
# end_feat = int(sys.argv[2])

base = "assisted_mutation/"
model = liblinearutil.load_model("Marvin/models/model_all_liblinear-L2")

def init_logger(thread_id):
    std_logger = logging.getLogger("standard_logger_" + str(thread_id))
    std_logger.setLevel(logging.DEBUG)
    std_fh = logging.FileHandler(base + "master_" + str(thread_id) + ".log")
    std_fh.setLevel(logging.DEBUG)
    std_logger.addHandler(std_fh)

    feature_logger = logging.getLogger("feature_logger_" + str(thread_id))
    feature_logger.setLevel(logging.DEBUG)
    feature_fh = logging.FileHandler(base + "features_" + str(thread_id) + ".log")
    feature_fh.setLevel(logging.DEBUG)
    feature_logger.addHandler(feature_fh)

    return std_logger, feature_logger

class DummyFile(object):
    def flush(self): pass
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

def evasion_thread(id, malicious_samples, evasive_output):
    std_logger, feature_logger = init_logger(id)

    evasive_file = open(evasive_output, "w+")

    for malicious_sample in malicious_samples:

        malicious = load_record(malicious_sample)
        malicious_orig = copy.deepcopy(malicious)

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

if __name__ == '__main__':
    with open("seeds/benign1.seed", "r") as f:
        benign = load_record(f.read())
    print("Total features: " + str(len(benign.features.keys())))
    benign_list = list(benign.features.keys())


    samples = [[] for i in range(8)]

    with open(sys.argv[1], "r") as f:
        for i, line in enumerate(f):
            samples[i%8].append(line)
        # malicious_samples = f.readlines()

    threads = [None] * 8
    with nostdout(): 
        for i in range(8):
            threads[i] = threading.Thread(target=evasion_thread, args=(i, samples[i], "evasive/" + str(i) + ".evasive"))
            threads[i].start()

        for i in range(8):
            threads[i].join()

    subprocess.call("./consolidate.sh")
