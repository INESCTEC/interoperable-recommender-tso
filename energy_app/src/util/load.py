import dill
import json


def load_dill_file(file_path):
    # Load the dictionary using dill
    with open(file_path, 'rb') as f:
        data = dill.load(f)
    return data


def load_json_file(file_path):
    # Load the dictionary using dill
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data
