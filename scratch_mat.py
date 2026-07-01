import scipy.io
import os
import json
import numpy as np

mat_file = '/Users/rajanahuja/aadu/neuromorphic-drive-monitoring/data/3mdad/Annotations 3MDAD/Annotations 3MDAD/RGB1S10AC1.mat'

def default_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

if os.path.exists(mat_file):
    try:
        data = scipy.io.loadmat(mat_file)
        clean_data = {k: v for k, v in data.items() if not k.startswith('__')}
        print(json.dumps(clean_data, default=default_serializer, indent=2))
    except Exception as e:
        print(f"Error reading mat file: {e}")
else:
    print("File not found")
