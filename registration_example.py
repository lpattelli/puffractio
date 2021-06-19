# xy registration example

import puffractio as pf
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

#%% generate Challenge and PUF

challenge_grid = 32 # grid size
upscale = 64        # magnification factor

challenge = pf.Challenge(challenge_grid)

Ngrid = challenge_grid * upscale
pixelsize = 0.1
pufsize = Ngrid * pixelsize

f = 0.50
rpart = 5
rexcl = 6

Npart = round(f*Ngrid**2 / np.pi / rexcl**2)
puf = pf.PUFmask(Ngrid, Npart, rpart, rexcl)

#%% propagate to a target off-axis position

wl = 0.5
target_xyz = [950, 650, 4000] # propagate at target coordinates
scaleupby = 4

R0 = pf.Response(challenge, puf, wavelength=wl, pixelsize=pixelsize)

speckle0 = R0.propagate(target_xyz[0], target_xyz[1], target_xyz[2], scaleupby=scaleupby)
speckle0.draw(reduce_matrix=[1,1])
clim = plt.gca().images[-1].get_clim()
plt.gca().images[-1].set_clim((0, clim[-1]))
plt.gca().set_title("original speckle")

#%% shift the puf and propagate again to the same position

puf.shift(101, -57) # this shift is defined in pixel units...

R1 = pf.Response(challenge, puf, wavelength=wl, pixelsize=pixelsize)
speckle1 = R1.propagate(target_xyz[0], target_xyz[1], target_xyz[2], scaleupby=scaleupby)
speckle1.draw(reduce_matrix=[1,1])
plt.gca().images[-1].set_clim((0, clim[-1])) # use same color axis
plt.gca().set_title("speckle after PUF shift")

#%% run speckle registration to find the matching propagation xy coordinates

f0 = np.abs(speckle0.u)**2 # reference speckle pattern
regx, regy = R1.registerxy(f0, plotfit=True, fitwidth=10) # also plotting the xcorr fit

speckle_reg = R1.propagate(target_xyz[0]-regx, target_xyz[1]-regy, target_xyz[2], scaleupby=scaleupby)
speckle_reg.draw(reduce_matrix=[1,1])
plt.gca().images[-1].set_clim((0, clim[-1]))
plt.gca().set_title("registered speckle")
