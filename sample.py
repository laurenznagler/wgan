import torch as th 
from models import ToyGenerator
import utils
import imageio.v3 as iio
import numpy as np

wgan = ToyGenerator(100)
z = utils.sample_noise(1, 100, 1, 1)

for eps in range(0, 101, 10):
    print(eps)
    wgan.load_state_dict(th.load(f"./weights/gen_{eps}.pt", map_location=th.device('cpu')))
    wgan.eval()
    generated = wgan(utils.sample_noise(1, 100, 1, 1)).squeeze().detach().cpu().numpy()
    iio.imwrite(f"./samples/evolution/sample_epoch_{eps}.png", (generated.clip(0, 1) * 255).astype(np.uint8))