import liblinearutil

class Model:

    model = None

    def load(self, model_file):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
    
    def predict(self, x, y):
        raise NotImplementedError

class LiblinearModel(Model):

    def load(self, model_file):
        model = liblinearutil.load_model(model_file)

    def predict(self, x, y):
        liblinearutil.predict(y, x, self.model, '-b 1')

class LiblinearL1Model(LiblinearModel):
    def train(self):
        model = liblinearutil.train(y, x, '-s 6 -n 8')

class LiblinearL2Model(LiblinearModel):
    def train(self):
        model = liblinearutil.train(y, x, '-s 0 -n 8')

class TorchModel(Model):

    def load(self, model_file):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
    
    def predict(self, x, y):
        raise NotImplementedError
