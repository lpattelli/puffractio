# example script

import puffractio as pf
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

#%% Challenges

challenge_grid = 32 # grid size
upscale = 128       # magnification factor

# only square PUFs are implemented at the moment
challenge = pf.Challenge(challenge_grid)

# show the generated challenge
challenge.show(upscale=upscale)

# save a png file with the upscaled challenge
# challenge.save('challenge.png', upscale=upscale)

# save another png of the same challenge with
# 2 macro-pixels flipped, in a different folder
# challenge.save('challenge_flip2.png', upscale=upscale, flip=2, path='./flipped')

#%% PUFs

Ngrid = challenge_grid * upscale
pixelsize = 0.1
pufsize = Ngrid * pixelsize

f = 0.53
rpart = 5
rexcl = 6

Npart = round(f*Ngrid**2 / np.pi / rexcl**2)

# generate, show and save a large PUF to file
puf = pf.PUFmask(Ngrid, Npart, rpart, rexcl)
puf.show()
# puf.save('puf.png')

# permanently remove 100 holes
# puf.remove_holes(100)
# puf.save('puf_m100.png')

# now try to put 1100 holes back in
# puf.add_holes(1100)
# puf.save('puf_p1100.png')

#%%

R = pf.Response(challenge, puf, wavelength=0.5, pixelsize=pixelsize)

speckle = R.propagate(200, 100, 20000)
# speckle.draw(reduce_matrix=[1,1])
ushrink = R.shrinkby(32)
ushrink.draw(reduce_matrix=[1,1])

# speckle = np.abs(speckle.u)**2
