import torch as th 
import numpy as np
import imageio.v3 as iio

import utils
from models import ToyDiscriminator, ToyGenerator

# Improved WGAN training - https://arxiv.org/pdf/1704.00028

# parameters 
latent_dim = 100
n_epochs = 100
n_critic = 3
batch_sz = 4
lr = 5e-4
lmbda_gp = 10.

# models
discriminator = ToyDiscriminator()
generator = ToyGenerator(input_size=latent_dim)

# optimizers
opt_gen = th.optim.Adam(generator.parameters(), lr=lr)
opt_disc = th.optim.Adam(discriminator.parameters(), lr=lr)

# prep data
x_train, x_test, masks = utils.loadl_mnist()
x_train = th.utils.data.TensorDataset(x_train)
dataloader = th.utils.data.DataLoader(x_train, batch_size=batch_sz, shuffle=True)

# training
for epoch in range(n_epochs + 1):
    discriminator.train()
    generator.train()
    
    if epoch % 10 == 0:
        th.save(discriminator.state_dict(), f"./weights/disc_{epoch}.pt")
        th.save(generator.state_dict(), f"./weights/gen_{epoch}.pt")
        
    for _, real_batch in enumerate(dataloader):
        real_batch = real_batch[0]
        # sample noise 
        z = utils.sample_noise(batch_sz, latent_dim, 1, 1)
        # generator
        fake_batch = generator(z)
        # discriminator
        logits_real = discriminator(real_batch)
        logits_fake = discriminator(fake_batch)
        
        # loss 
        loss = th.mean(logits_fake) - th.mean(logits_real)
        
        # gradient penalty
        eps = th.rand(batch_sz, 1, 1, 1)
        xhat = eps * fake_batch + (1. - eps) * real_batch

        xhat_discr = discriminator(xhat)
        grads = th.autograd.grad(xhat_discr, xhat, grad_outputs=th.ones_like(xhat_discr), create_graph=True)[0]
        grad_penalty = ((th.sqrt((grads ** 2).sum(dim=(1, 2, 3))) - 1) ** 2).mean()
        loss = loss + lmbda_gp * grad_penalty
        
        if _ % 100 == 0:
            print(f"Discriminator Loss in Epoch {epoch} and Iter {_}: {-loss}")
            
        # update discriminator
        opt_disc.zero_grad()
        loss.backward()
        opt_disc.step()
        
        if _ % n_critic == 0:
            z = utils.sample_noise(batch_sz, latent_dim, 1, 1)
            loss = (-1)*th.mean(discriminator(generator(z)))
            if _ % (100 * n_critic) == 0:
                print(f"Generator Loss in Epoch {epoch} and Iter {_}: {loss}")
            opt_gen.zero_grad()
            loss.backward()
            opt_gen.step()
            
    # log a sample in each epoch
    generator.eval()
    generated = generator(utils.sample_noise(1, 100, 1, 1)).squeeze().detach().cpu().numpy()
    iio.imwrite(f"./samples/sample_epoch_{epoch}.png", (generated.clip(0, 1) * 255).astype(np.uint8))
    
    