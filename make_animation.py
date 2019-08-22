import sys, time, os
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
from shutil import copyfile




inDir = 'image_output/'
outDir = '/home/bruno/Desktop/'

image_name = 'image'

out_anim_name = 'dm_50Mpc_3D'

cmd = 'ffmpeg -framerate 60  '
# cmd += ' -start_number 45'
cmd += ' -i {0}{1}_%d.png '.format( inDir, image_name )
cmd += '-pix_fmt yuv420p '
# cmd += '-b 50000k '
cmd += '{0}{1}.mp4'.format( outDir, out_anim_name )
cmd += ' -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"'
cmd += ' -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2"'
os.system( cmd )

