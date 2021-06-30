import json
import numpy as np

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def save_json(file_path, dict):
    f = open(file_path, 'w')
    json.dump(dict, f, cls=MyEncoder)
    f.close()

def load_json(file_path):
    f = open(file_path, 'r')
    return json.load(f)

