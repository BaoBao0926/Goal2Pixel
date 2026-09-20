import gzip
import json
import pickle

def write_json(data, path):
    with open(path, "w") as file:
        file.write(json.dumps(data))


def load_json(path):
    file = open(path, "r")
    data = json.loads(file.read())
    return data

def load_dataset(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        data = json.load(file)  # Use json.load for reading JSON directly
    return data

def write_dataset(data, path):
    with gzip.open(path, "wt") as f:
        json.dump(data, f)

def save_pickle(data, path):
    file = open(path, "wb")
    data = pickle.dump(data, file)


def load_pickle(path):
    file = open(path, "rb")
    data = pickle.load(file)
    return data