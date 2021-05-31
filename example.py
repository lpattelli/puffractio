# example script

import puffractio as pf
import numpy as np

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

Ngrid = challenge_grid * upscale # make a PUF as large as the magnified Challenge

f = 0.53
rpart = 5
rexcl = 6

Npart = round(f*Ngrid**2 / np.pi / rexcl**2)

# generate, show and save a large PUF to file
puf = pf.PUFmask(Ngrid, Npart, rpart, rexcl)
puf.show()
# puf.save('puf.png')

# permanently remove 100 holes. the PUF now has only 999900 holes
# puf.remove_holes(100)
# puf.save('puf_m100.png')

# try to add 1100 holes, reaching 101000
# puf.add_holes(1100)
# puf.save('puf_p1100.png')

#%%

R = pf.Response(challenge, puf, wavelength=0.5, pixelsize=0.1)

speckle = R.propagate(200, 100, 20000)
# speckle.draw()
ushrink = R.shrinkby(32)
ushrink.draw()

# speckle = np.abs(speckle.u)**2
