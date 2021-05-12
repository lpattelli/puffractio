##PUF generation
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from skimage import draw


def edge_particles(p, r):
    """ expects particle positions in a [0,1] square """
    p = np.concatenate([p + [ 1, 0], p + [ 1, 1], p + [ 0, 1], p + [-1, 1],
                        p + [-1, 0], p + [-1,-1], p + [ 0,-1], p + [ 1,-1]])
    p = np.delete(p, np.any(p > 1+r, 1), 0)
    p = np.delete(p, np.any(p <  -r, 1), 0)
    return p


# def remove_overlapping(pnew, r, p=None):
#     """ remove overlapping particles in a group of particles pnew or with
#         respect to a group of already placed particles p"""
#     ovlp = np.triu(cdist(pnew, pnew) < 2*r, 1) if (p is None) else cdist(pnew, p) < 2*r
#     return pnew[np.sum(ovlp, 1) == 0]

def remove_overlapping(pnew, r, p=None, boxsize=None):
    kdt1 = KDTree(pnew, boxsize=boxsize)
    kdt2 = KDTree(pnew, boxsize=boxsize)
    if p is not None:
        kdt2 = KDTree(p, boxsize=boxsize)
    dm = kdt1.sparse_distance_matrix(kdt2, 2*r).nonzero()
    return np.delete(pnew, dm[0], 0)


# def RSA_2D(N, r, Ngrid, p=None, seed=None):
#     np.random.seed(seed)
#     r /= Ngrid # go to normalized units
#     p = np.random.rand(1, 2) if (p is None) else p/Ngrid
#     with tqdm(initial=np.size(p, 0), total=N, desc="packing disks") as pbar:
#         while np.size(p, 0) < N:
#             pnew = np.random.rand(N, 2)
#             pnew = remove_overlapping(pnew, r)
#             pnew = remove_overlapping(pnew, r, p)
#             p = np.concatenate([p, pnew], 0)
#             pbar.n = min(N, np.size(p,0)); pbar.refresh()
#     return Ngrid*p[:N,]


def RSA_2D_PBC(N, r, Ngrid, p=None, seed=None):
    np.random.seed(seed)
    r /= Ngrid # go to normalized units
    p = np.random.rand(1, 2) if (p is None) else p/Ngrid
    with tqdm(initial=np.size(p, 0), total=N, desc="packing disks") as pbar:
        while np.size(p, 0) < N:
            pnew = np.random.rand(N, 2)
            pnew = remove_overlapping(pnew, r)
            pnew = remove_overlapping(pnew, r, p)
            pedg = edge_particles(np.concatenate([p,pnew]), r)
            pnew = remove_overlapping(pnew, r, pedg)
            p = np.concatenate([p, pnew], 0)
            pedg = edge_particles(p, r)
            p = remove_overlapping(p, r, pedg)
            pbar.n = min(N, np.size(p,0)); pbar.refresh()
    return Ngrid*p[:N,]


def RSA_2D_PBC_new(N, r, Ngrid, p=None, seed=None):
    np.random.seed(seed)
    r /= Ngrid # go to normalized units
    p = np.random.rand(1, 2) if (p is None) else p/Ngrid
    with tqdm(initial=np.size(p, 0), total=N, desc="packing disks") as pbar:
        while np.size(p, 0) < N:
            pnew = np.random.rand(N, 2)
            pnew = remove_overlapping(pnew, r, p, boxsize=1)
            p = remove_overlapping(np.concatenate([p,pnew]), r, boxsize=1)
            pbar.n = min(N, np.size(p,0)); pbar.refresh()
    return Ngrid*p[:N,]


# def draw_mask(p, Ngrid, r):
#     x,y = np.mgrid[:Ngrid, :Ngrid]
#     m = np.zeros_like(x)
#     for pidx in tqdm(range(np.size(p,0)), desc="drawing disks"):
#         m += (x-p[pidx,0])**2 + (y-p[pidx,1])**2 < r**2
#     return m


# def draw_mask_PBC(p, Ngrid, r):
#     """ will draw also the extra particles due to the application of PBC """
#     x,y = np.mgrid[:Ngrid, :Ngrid]
#     p = np.concatenate([p, Ngrid*edge_particles(p/Ngrid, r/Ngrid)])
#     m = np.zeros_like(x)
#     for pidx in tqdm(range(np.size(p,0)), desc="drawing disks"):
#         m += (x - p[pidx,0])**2 + (y - p[pidx,1])**2 < r**2
#     return m


def draw_mask_PBC(p, Ngrid, r):
    """ will draw also the extra particles due to the application of PBC """
    p = np.concatenate([p, Ngrid*edge_particles(p/Ngrid, r/Ngrid)])
    m = np.zeros((Ngrid,Ngrid), dtype=bool)
    for pidx in tqdm(range(np.size(p,0)), desc="drawing disks"):
        m[draw.disk(p[pidx,], r, shape=m.shape)] = True
    return m


Ngrid = 1024*4
npart = 20000
rexcl = 12.0
rpart = 10.0

p = RSA_2D_PBC_new(npart, rexcl, Ngrid)
m = 1 - draw_mask_PBC(p, Ngrid, rpart)

nominalpf = npart*np.pi*rpart**2 / Ngrid**2
actualpf  = np.sum(1-m)/m.size # may be slightly different due to pixelization

fig, ax = plt.subplots()
ax.imshow(m, plt.cm.Greys)
plt.show()

# tile the final mask a few times to check that PBC apply correctly
fig, ax = plt.subplots()
plt.imshow(np.tile(m, [2,2]), plt.cm.Greys)
plt.show()
