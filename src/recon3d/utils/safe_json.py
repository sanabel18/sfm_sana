import json
import numpy as np


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def safe_json_dump(obj, fp):
    json.dump(obj, fp, cls=SafeJSONEncoder)

