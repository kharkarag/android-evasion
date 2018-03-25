import ast
import os


class Sample:
    
    def __init__(self, feature_type):
        self.label = 0
        self.score = 0
        self.sample_id = 0
        self.features = feature_type()
        self.features.clear()
        self.added_feat = feature_type()
        self.added_feat.clear()
        self.cost = 0
        self.fitness = 1

    def stringify(self):
        return_string = "Sample: " + str(self.sample_id)
        return_string += " | Label: " + str(self.label)
        return_string += " | Score: " + str(self.score)
        return_string += "\n" + str(self.features)
        return return_string


class MarvinSample(Sample):

    def __init__(self):
        super(MarvinSample, self).__init__(dict)
        self.perm_added = 0
        self.static_added = 0
        self.dynamic_added = 0

    def sparse_arff(self):
        return_string = "+1" if self.label == 1 else "-1"
        for key in sorted(self.features.keys()):
            return_string += " " + str(key) + ":" + str(self.features[key])
        return_string += "\n"
        return return_string


class OpseqSample(Sample):
    
    def __init__(self):
        super(OpseqSample, self).__init__(list)

    def opcode_sequence(self):
        return_string = ""
        for feat in self.features:
            return_string += str(feat) + "\n"
        return return_string


def load_marvin(line):
    sample = MarvinSample()
    sample.label = int(line[0:2])
    line = line[3:]
    sample.features = ast.literal_eval("{" + ", ".join(line.split(" ")) + "}")
    del sample.features[1]
    return sample


def load_seed(line):
    sample = MarvinSample()
    sample.label = int(line[0:2])
    line = line[3:]
    sample.features = ast.literal_eval("{" + ", ".join(line.split(" ")) + "}")
    return sample


def load_opseq(sequence, label, filename):
    sample = OpseqSample()
    sample.label = label
    sample.features = sequence
    sample.sample_id = os.path.splitext(filename)[0]
    return sample


if __name__ == "__main__":
    pass
