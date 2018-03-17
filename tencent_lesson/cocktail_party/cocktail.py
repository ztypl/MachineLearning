# coding : utf-8
# create by ztypl on 2017/9/7

from mdp import fastica
from scikits.audiolab import flacread, flacwrite
from numpy import abs, max

# Load in the stereo file
recording, fs, enc = flacread('Mixed3.flac')

# Perform FastICA algorithm on the two channels
sources = fastica(recording)

# The output levels of this algorithm are arbitrary, so normalize them to 1.0.
sources /= max(abs(sources), axis=0)

source1 = sources[:,0]
source2 = sources[:,1]

# Write back to a file
flacwrite(source1, 'source21.flac', fs, enc)
flacwrite(source2, 'source22.flac', fs, enc)