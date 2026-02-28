import numpy as np
import os
import urllib.request
import gzip

def load_mnist(path=None):
    if path is None:
        path = 'data/mnist'

    if not os.path.exists(path):
        os.makedirs(path)

    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    ]

    for url in urls:
        filename = url.split('/')[-1]
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            # Note: Some servers block simple urllib requests, but let's try.
            # In a real scenario, we'd handle potential download failures.
            try:
                urllib.request.urlretrieve(url, filepath)
            except:
                print(f"Failed to download {filename}. Please provide MNIST files manually in {path}")
                return None, None, None, None

    def read_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

    def read_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    x_train = read_images(os.path.join(path, 'train-images-idx3-ubyte.gz'))
    y_train = read_labels(os.path.join(path, 'train-labels-idx1-ubyte.gz'))
    x_test = read_images(os.path.join(path, 't10k-images-idx3-ubyte.gz'))
    y_test = read_labels(os.path.join(path, 't10k-labels-idx1-ubyte.gz'))

    return (x_train, y_train), (x_test, y_test)
