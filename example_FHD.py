#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example script showing how to binarize and compute the fractional
hamming distance (FHD) distribution for a given PUF using puffractio
"""

import puffractio as pf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.spatial.distance import hamming
from scipy.optimize import curve_fit

plt.close('all')

#%% generate PUF

challenge_grid = 16 # grid size
upscale = 64       # size of each macro-pixel

Ngrid = challenge_grid * upscale
pixelsize = 0.1                 # physical size of a pixel, in µm
pufsize = Ngrid * pixelsize     # physical size of the PUF

f = 0.5     # hole packing fraction = 50%
rpart = 4   # actual hole radius in pixel
rexcl = 5   # exclusion hole radius in pixel

Npart = round(f*Ngrid**2 / np.pi / rexcl**2)
puf = pf.PUFmask(Ngrid, Npart, rpart, rexcl, seed=42)

#%% obtain array of responses for NCH challenges

NCH = 25
wl = 0.632
target_xyz = [400, 0, 1000] # propagate at target coordinates off-axis

scaleupby = 4   # pixelsize multiplier at the target distance
                # the number of pixels is still the same,
                # but their physical area is scaleupby**2 times larger
                
downscale = 8   # reduce the final response size by this factor

keys = np.zeros((int(Ngrid/downscale)**2, NCH))
Rarr = np.zeros((int(Ngrid/downscale), int(Ngrid/downscale), NCH))

lambd = 2       # Gabor filter wavelength, in pixels
theta = np.pi/4 # Gabor filter angle, in radians

for i in tqdm(range(NCH), desc="Evaluating CRPs"): 

    challenge = pf.Challenge(challenge_grid, seed=i)
    #challenge.save(f'challenge_{i}.png', upscale=upscale, path='./challenges')
    R0 = pf.Response(challenge, puf, wavelength=wl, pixelsize=pixelsize)
    
    speckle0 = R0.propagate(target_xyz[0], target_xyz[1], target_xyz[2], scaleupby=scaleupby, verbose=False)
    #speckle0.draw(reduce_matrix=[1,1])
    
    urescaled = R0.shrinkby(downscale)
    mag, phase = R0.gaborfilter(urescaled, lambd, theta)
    rh = np.imag(mag * np.exp(1j * phase)) # the imaginary part is odd
    rhbw = rh >= 0.
    Rarr[:,:,i] = urescaled.u
    keys[:,i] = rhbw.flatten()

#%% normalize responses

avgR = np.mean(Rarr,2)
normRarr = Rarr / np.repeat(avgR[:,:,np.newaxis], NCH, axis=2)

for i in tqdm(range(NCH), desc="Hash and binarize"): 
    urescaled.u = normRarr[:,:,i] # reuse/overwrite this previous object for convenience
    mag, phase = R0.gaborfilter(urescaled, lambd, theta)
    rh = np.imag(mag * np.exp(1j * phase)) # sine is an odd function
    rhbw = rh >= 0.
    keys[:,i] = rhbw.flatten()

#%% compute the fractional hamming distance

unlike = np.zeros((NCH,NCH))

for i in tqdm(range (NCH), desc="Evaluating FHD"):
    for j in range (i, NCH):
        unlike[i,j] = hamming(keys[:,i], keys[:,j]);

#%% Fit 

def gaussfunc(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

data = unlike[np.triu_indices_from(unlike)]

counts, bins = np.histogram(data, bins=100)
hist, bin_edges = np.histogram(data, density=True, bins=bins)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

coeff, var_matrix = curve_fit(gaussfunc, bin_centres, hist, p0=[1., 0., 1.])

hist_fit = gaussfunc(bin_centres, *coeff)

plt.hist(bins[:-1], bins, weights=counts, density= True, alpha=0.7)
plt.plot(bin_centres, hist_fit, 'b--', linewidth=2)

plt.xlabel('FHD')
plt.ylabel('counts')
plt.title(f'FHD histogram: $\mu={coeff[1]:.3f}$, $\sigma={np.abs(coeff[2]):.3f}$')
plt.grid(True)

plt.show()

# plt.imshow(unlike, cmap='GnBu')
# plt.xlabel('$MF_j$')
# plt.ylabel('$MF_i$')
# plt.title('FHD map distribution')
# cax = plt.axes([1, 0.1, 0.1, 0.8])
# plt.colorbar(cax=cax)
# plt.show()
