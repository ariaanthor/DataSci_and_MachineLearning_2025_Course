import numpy as np

def distance(p, q):
    return np.sqrt((p - q) @ (p - q))