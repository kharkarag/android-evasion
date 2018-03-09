import ast

class Record:
    label = 0
    features = dict()
    score = 0
    record_id = 0
    added_feat = 0

    def stringify(self):
        return_string = "Record: " + str(self.record_id)
        return_string += " | Label: " + str(self.label)
        return_string += " | Score: " + str(self.score)
        return_string += "\n" + str(self.features)
        return return_string

    def sparse_arff(self):
        return_string = "+1" if self.label == 1 else "-1"
        for key in sorted(self.features.keys()):
            return_string += " " + str(key) + ":" + str(self.features[key])
        return_string += "\n"
        return return_string

def load_record(line):
    record = Record()
    record.label = int(line[0:2])
    line = line[3:]
    record.features = ast.literal_eval("{" + ", ".join(line.split(" ")) + "}")
    del record.features[1]
    return record

def load_seed(line):
    record = Record()
    record.label = int(line[0:2])
    line = line[3:]
    record.features = ast.literal_eval("{" + ", ".join(line.split(" ")) + "}")
    return record

if __name__ == "__main__":
    pass