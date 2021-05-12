import matplotlib.pyplot as plt
from scipy import sparse
import numpy as np
import PIL
import os


def gen_challenge(p, q, f=0.5, exact=True, seed=None):
    np.random.seed(seed)
    if exact:
        challenge = np.full(p*q, False)
        challenge[:int(p*q*f)] = True
        np.random.shuffle(challenge)
    else:
        challenge = np.random.choice([True, False], size=p*q, p=[f, 1-f])
    return challenge.reshape((p,q))

def upscale_challenge(c, n, m, save=False, fname=None, savepath='.'):
    d1, d2 = n//c.shape[0], m//c.shape[1]
    c = sparse.kron(c, np.ones((d1, d2))).toarray()
    c = np.pad(c, ((0,n-c.shape[0]),(0,m-c.shape[1])), mode='edge')
    if save:
        img = PIL.Image.fromarray(c*255).convert("1")
        img.save(os.path.join(savepath, fname), optimize=True)
    return c

def flip_pixel(c, N=1, seed=None):
    np.random.seed(seed)
    f = c.copy()
    mask = np.full(c.size, False)
    mask[:N] = True
    np.random.shuffle(mask)
    mask = mask.reshape(c.shape)
    f[mask] = np.logical_not(f[mask])
    return f


p, q = 16, 16
n, m = 512, 512

# plot example
c = gen_challenge(p, q)
cflipped = flip_pixel(c)

c = upscale_challenge(c, n, m)
cflipped = upscale_challenge(cflipped, n, m)

fig, axarr = plt.subplots(1,3)
axarr[0].imshow(c)
axarr[1].imshow(cflipped)
axarr[2].imshow(c.astype(int)-cflipped.astype(int))

#%%

for realiz in range(10):
    c = gen_challenge(p, q, seed=realiz)
    cflipped = flip_pixel(c, seed=realiz)

    c = upscale_challenge(c, n, m, save=True, fname='challenge_%d.png'%realiz)
    cflipped = upscale_challenge(cflipped, n, m, save=True, fname='challenge_1flip_%d.png'%realiz)

