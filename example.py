# example script

import puffractio as pf


#%% Challenges

challenge_grid = 50 # grid size
upscale = 100       # magnification factor

# only square PUFs are implemented at the moment
challenge = pf.Challenge(challenge_grid)

# show the generated challenge
challenge.show(upscale=upscale)

# save a png file with the upscaled challenge
challenge.save('challenge.png', upscale=upscale)

# save another png of the same challenge with
# 2 macro-pixels flipped, in a different folder
challenge.save('challenge_flip2.png', upscale=upscale, flip=2, path='./flipped')

#%% PUFs

Ngrid = challenge_grid * upscale # make a PUF as large as the magnified Challenge
Npart = 100000
rpart = 5
rexcl = 6

# generate, show and save a large PUF to file
puf = pf.PUFmask(Ngrid, Npart, rpart, rexcl)
puf.show()
puf.save('puf.png')

# permanently remove 100 holes. the PUF now has only 999900 holes
puf.remove_holes(100)
puf.save('puf_m100.png')

# try to add 1100 holes, reaching 101000
puf.add_holes(1100)
puf.save('puf_p1100.png')