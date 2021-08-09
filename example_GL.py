# example script

import sys
sys.path.append("/Users/giuseppeemanuelelio/Documents/GitHub/puffractio")

import puffractio_GL as pf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import savemat
from scipy.spatial.distance import hamming
from scipy.optimize import curve_fit

plt.close('all')

#%% generate Challenge and PUF

challenge_grid = 16 # grid size
upscale = 32      # magnification factor

Ngrid = challenge_grid * upscale
pixelsize = 0.1
pufsize = Ngrid * pixelsize
NCH=50

f = 0.5
rpart = 4
rexcl = 5

Npart = round(f*Ngrid**2 / np.pi / rexcl**2)
puf = pf.PUFmask(Ngrid, Npart, rpart, rexcl,seed=1000)
#puf.save('puf_00.png',path='./Example/PUFs')
#%%
wl = 0.63
target_xyz = [-500, 0, 1000] # propagate at target coordinates off-axis
scaleupby = 2
Shrinpar=4
Reduction=int(Ngrid/Shrinpar)

f0=np.zeros((Ngrid,Ngrid,NCH))
a=np.zeros((Reduction,Reduction,NCH))
#%%Evaluation of the first response PUF is in XY(0,0)
for i in tqdm(range(NCH), desc="Evaluating CRPs"): 

    challenge = pf.Challenge(challenge_grid,seed=i)
    #challenge.save('CR_%d.png'%i, upscale=upscale, path='./Example/Challenges')
    R0 = pf.Response(challenge, puf, wavelength=wl, pixelsize=pixelsize)
    
    speckle0 = R0.propagate(target_xyz[0], target_xyz[1], target_xyz[2], scaleupby=scaleupby, verbose=False)
    #speckle0.draw(reduce_matrix=[1,1])
    #clim = plt.gca().images[-1].get_clim()
    # plt.gca().images[-1].set_clim((0, clim[-1]))
    # plt.gca().set_title("original speckle")
    
    ushrink = R0.shrinkby(Shrinpar)
    #ushrink.draw(reduce_matrix=[1,1])
    #clim = plt.gca().images[-1].get_clim()
    ushrink= np.abs(ushrink.u)**2
    a[:,:,i]= ushrink

a*=2**15
a=a.astype('uint16')
mdic = {"a": a}
#savemat("./Example/CRPs.mat", mdic)

#%%Image Gabor Hash 

Tout_a=np.mean(a,2);
df_list=[]
MF=np.array([])
wavelength=6  #Filter wavelength px
orientation=45 #Filter angle orientation
for i in tqdm(range(NCH), desc="Gabor hashing images"):
    image=a[:,:,i]
    image=image/Tout_a
    mag, phase = pf.FHD.gabor(image, wavelength=wavelength, orientation=orientation)
    res=mag*np.cos(phase)
    res=np.abs(res/np.max(res))
    res= (res >= 0.06).astype(int)
    A = np.asarray(res).reshape(-1)
    ll=np.size(A)
    df_list += [A]

MF=np.vstack(df_list)
Unlike=np.zeros((NCH,NCH))

for i in tqdm(range (NCH), desc="Unlike evaluation"):
    for j in range (NCH):
        Unlike[i,j]= hamming(MF[i,:], MF[j,:]);
        


#%% Fit 
data=Unlike

counts, bins = np.histogram(Unlike, bins=100)
hist, bin_edges = np.histogram(data, density=True, bins=bins)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

p0 = [1., 0., 1.] # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)

coeff, var_matrix = curve_fit(pf.FHD.gauss, bin_centres, hist, p0=p0)

hist_fit = pf.FHD.gauss(bin_centres, *coeff) # Get the fitted curve

plt.hist(bins[:-1], bins, weights=counts, density= True, alpha=0.7)
plt.plot(bin_centres, hist_fit, 'b--', linewidth=2)

# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
# print ('Fitted mean = ', coeff[1])
# print ('Fitted standard deviation = ', coeff[2])
#You can also display the parameters on the graph title

plt.xlabel('FHD')
plt.ylabel('Counts')
plt.title(r'$\mathrm{Histogram\ of\ FHD:}\ \mu=%.3f,\ \sigma=%.3f$' %(coeff[1],coeff[2]))
plt.grid(True)

plt.show()

plt.imshow(Unlike, cmap='GnBu')
plt.xlabel('$MF_j$')
plt.ylabel('$MF_i$')
plt.title(r'$\mathrm{FHD\ Map\ Distribution:}$ ')
cax = plt.axes([1, 0.1, 0.1, 0.8])
plt.colorbar(cax=cax)
plt.show()
