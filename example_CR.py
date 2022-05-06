#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example script showing how to generate a 2D PUF mask,
illuminate it with a challenge and get a speckle response
"""

import puffractio as pf
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

#%% define challenge

challenge_grid = 16 # grid size
upscale = 128       # size of each macro-pixel

# only square PUFs are implemented at the moment
challenge = pf.Challenge(challenge_grid)

# show the generated challenge upscaled to its actual size
challenge.show(upscale=upscale)

# save a png file with the upscaled challenge
# challenge.save('challenge.png', upscale=upscale)

#%% create PUF

Ngrid = challenge_grid * upscale
pixelsize = 0.1                 # physical size of a pixel, in µm
pufsize = Ngrid * pixelsize     # physical size of the PUF

f = 0.5     # hole packing fraction = 50%
rpart = 5   # actual hole radius in pixel
rexcl = 6   # exclusion hole radius in pixel

# corresponding number of holes in the PUF mask
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

#%% propagate field to the desired image plane

R = pf.Response(challenge, puf, wavelength=0.5, pixelsize=pixelsize)

# the speckle pattern will be as large as the PUF (Ngrid x Ngrid)
# and is calculated at a distance z, around the (x,y) position
# the PUF mask is assumed to be centered in (0,0,0)
speckle = R.propagate(x=400, y=200, z=2000)
speckle.draw(reduce_matrix=[1,1])

# ushrink = R.shrinkby(32)
# ushrink.draw(reduce_matrix=[1,1])

