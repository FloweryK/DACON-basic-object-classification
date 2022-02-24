import os
import json
import pickle

def save(data, path):
    # make proper directories if needed.
    path_split = list(filter(None, path.split('/')))
    if len(path_split) > 1:
        dir = '/'.join(path_split[:-1])
        path = '/'.join(path_split)

        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

    # if saving json, use json dump.
    if path[-4:] == 'json':
        with open(path, 'w', encoding='UTF8') as f:
            json.dump(data, f, ensure_ascii=False)
    # if not, use pickle dump.
    else:
        with open(path, 'wb') as f:
            pickle.dump(data, f)


def load(path):
    # if loading json, use json load
    if path[-4:] == 'json':
        with open(path, encoding='UTF8') as f:
            result = json.load(f)
    # if not, use pickle load.
    else:
        with open(path, 'rb') as f:
            result = pickle.load(f)

    return result