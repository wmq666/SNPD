import os
import scipy
import numpy as np


def make_path(*your_path):
    path = os.path.join(*your_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def tensor_to_numpy(x):
    x = x.detach().cpu().numpy()
    return x


def load_data(your_path, key):
    raw = scipy.io.loadmat(your_path)
    if key in raw:
        return raw[key]
    else:
        raise KeyError(f"The keyword '{key}' not exist。")


def norm(x):
    out = (x - min(x)) / (max(x) - min(x))
    return out.reshape(1024)


def SNR(gt, s):
    gt = gt.reshape(1024, )
    s = s.reshape(1024, )
    snr = 10 * np.log10(np.sum(gt ** 2) / np.sum((s - gt) ** 2))
    return snr


def SAD(gt, s):
    gt = gt.reshape(1024, )
    s = s.reshape(1024, )
    arc_cos = np.arccos(np.dot(gt, s) / (np.linalg.norm(gt, ord=2) * np.linalg.norm(s, ord=2)))
    sad = 1 - ((2 * arc_cos) / np.pi)
    return sad
