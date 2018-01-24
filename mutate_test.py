import sys
import contextlib
import subprocess
import copy
import logging
from util.util import *
from lib import liblinearutil

model = liblinearutil.load_model("Marvin/models/model_all_liblinear-L2")

std_logger = logging.getLogger("standard_logger")
std_logger.setLevel(logging.DEBUG)
std_fh = logging.FileHandler("logs/master.log")
std_fh.setLevel(logging.DEBUG)
std_logger.addHandler(std_fh)

feature_logger = logging.getLogger("feature_logger")
feature_logger.setLevel(logging.DEBUG)
feature_fh = logging.FileHandler("logs/features.log")
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

with open("seeds/benign1.seed", "r") as f:
    benign = load_record(f.read())
print("Total features: " + str(len(benign.features.keys())))
benign_list = list(benign.features.keys())

with open(sys.argv[1], "r") as f:
    malicious_samples = f.readlines()

for malicious_sample in malicious_samples:

    malicious = load_record(malicious_sample)
    malicious_orig = copy.deepcopy(malicious)

    best_probs = [1.0, 0.0]
    best_feat = -1

    std_logger.debug("------------------------------")
    print("------------------------------")
    std_logger.info(malicious_orig.stringify())
    for i in range(15):

        for feat in benign_list:
            malicious.features[feat] = 1

            with nostdout():
                p_labs, p_acc, p_vals = liblinearutil.predict([malicious.label], [malicious.features], model, '-b 1')
            score = p_vals[0][0]
            inv_score = p_vals[0][1]

            if best_probs[0] > score:
                    best_probs = [score, inv_score]
                    best_feat = feat

            malicious = copy.deepcopy(malicious_orig)

        # print("Feat: " + str(best_feat) + ", prob: " + str(best_probs))
        std_logger.info(str(best_feat) + ": " + str(best_probs))
        feature_logger.info(str(best_feat))
        malicious_orig.features[best_feat] = 1
        malicious = copy.deepcopy(malicious_orig)

    if best_probs[0] < 0.5:
        std_logger.warning("Success - final prob: " + str(best_probs[0]))
        print("Success - final prob: " + str(best_probs[0]))
