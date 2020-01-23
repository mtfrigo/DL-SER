import json

class Testset(object):

    def __init__(self, path):
        self.path = path
        self.dataset = None

        self.load()


    def load(self):

        with open(self.path) as json_file:
            self.dataset = json.load(json_file)

class Devset(object):

    def __init__(self, path):
        self.path = path
        self.dataset = None

        self.load()

    def load(self):

        with open(self.path) as json_file:
            self.dataset = json.load(json_file)

    def write(self, path, data):
        with open(path, 'w') as outfile:
            json.dump(data, outfile)


class DatasetHandler(object):

    def __init__(self, dataset):
        self.dataset = dataset

        # --- JSON
        self.ids = []
        self.activations = []
        self.valences = []
        self.features = [] 
        self.data = []
        self.labels = []

        self.parse()

    def parse(self):

        for p in self.dataset:
            self.dataset[p]['id'] = p # -- Write id to object

            self.ids.extend([p])
            self.activations.extend([self.dataset[p]['activation']])
            self.valences.extend([self.dataset[p]['valence']])
            self.features.extend([self.dataset[p]['features']])


            self.labels.extend([self.getLabel(self.dataset[p]['activation'], self.dataset[p]['valence'])])
            self.data.append(self.dataset[p])

    def getLabel(self, activation, valence):
        if activation == 0 and valence == 0: return 0
        elif activation == 1 and valence == 0: return 1
        elif activation == 0 and valence == 1: return 2
        else: return 3