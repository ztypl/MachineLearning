# coding : utf-8
# create by ztypl on 2017/9/7


from mdp import fastica
from scikits.audiolab import flacread, flacwrite
from numpy import array

sig1, fs1, enc1 = flacread('file1.flac')
sig2, fs2, enc2 = flacread('file2.flac')

sig1 = sig1[:,0]
sig2 = sig2[:,0]

minlen = min(len(sig1), len(sig2))

sig1 = sig1[:minlen]
sig2 = sig2[:minlen]

mixed1 = sig1 + 0.5 * sig2
mixed2 = sig1 + 0.8 * sig2

mixed = array([mixed1, mixed2]).T

flacwrite(mixed,'Mixed2.flac')
