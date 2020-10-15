import sys, time, os
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
from shutil import copyfile




# inDir = '/home/bruno/Desktop/ssd_0/data/cosmo_sims/1024_hydro_50Mpc/snapshots_pchw18/render_gas_temperature_new/'
inDir = '/home/bruno/Desktop/ssd_0/data/cosmo_sims/1024_hydro_50Mpc/snapshots_pchw18/render_dm_gas/'
outDir = '/home/bruno/Desktop/'


image_name = 'image'

out_anim_name = 'dm_gas_density_50Mpc_new'

cmd = 'ffmpeg -framerate 60  '
# cmd += ' -start_number 45'
cmd += ' -i {0}{1}_%d.png '.format( inDir, image_name )
cmd += '-pix_fmt yuv420p '
# cmd += '-b 50000k '
cmd += '{0}{1}.mp4'.format( outDir, out_anim_name )
# cmd += ' -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"'
# cmd += ' -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2"'
os.system( cmd )

