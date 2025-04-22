import numpy as np
import torch as th

def loadl_mnist():
    with np.load("./data/mnist/data.npz") as f:
        x_train = f["train_data"]
        x_test = f["test_data"]
        mask = f["mask"]
    return th.from_numpy(x_train).to(th.float).unsqueeze(1), \
           th.from_numpy(x_test).to(th.float).unsqueeze(1), \
           th.from_numpy(mask).to(th.float).unsqueeze(1)
           
def sample_noise(batch_size, *channels):
    return th.randn(batch_size, *channels).float()