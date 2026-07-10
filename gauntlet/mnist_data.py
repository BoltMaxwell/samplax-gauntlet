"""MNIST idx loading (raw pixel floats; rows apply their own normalization)."""

import gzip
import os
import struct
import urllib.request

import numpy as np

MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)
FILES = {
    "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
}


def download(root):
    os.makedirs(root, exist_ok=True)
    for img, lbl in FILES.values():
        for name in (img, lbl):
            path = os.path.join(root, name)
            if os.path.exists(path):
                continue
            for mirror in MIRRORS:
                try:
                    with urllib.request.urlopen(mirror + name + ".gz") as r:
                        data = gzip.decompress(r.read())
                    with open(path, "wb") as f:
                        f.write(data)
                    break
                except Exception:
                    continue


def _load(root, split):
    img_name, lbl_name = FILES[split]
    with open(os.path.join(root, lbl_name), "rb") as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)
    with open(os.path.join(root, img_name), "rb") as f:
        _, size, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows * cols)
    return images.astype(np.float32), labels


def load_mnist(root):
    download(root)
    return _load(root, "train"), _load(root, "test")
