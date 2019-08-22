import os, sys
import numpy as np
import h5py as h5

#Add Modules from other directories
currentDirectory = os.getcwd()
dataDirectory = currentDirectory + "/data_src/"
sys.path.extend([  dataDirectory ] )
from tools import *
from load_data_cholla import load_snapshot_data_particles, load_snapshot_data_grid

dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
# dataDir = '/raid/bruno/data/'
# inDir = dataDir + 'cosmo_sims/cholla_pm/256_cool_uv_50Mpc/data_PPMC_HLLC_SIMPLE_eta0.001_0.0400/'
inDir = dataDir + 'cosmo_sims/cholla_pm/128_cool/data_float32/'
enzoDir = inDir
outDir = inDir


# fileKey = 'particles'
# fileKey = 'grid'

fields_grid = ['density', 'temperature']
fields_particles = ['density']

fileKeys = [ 'particles', 'grid']

for fileKey in fileKeys:

  outFileName = 'stats_{0}.h5'.format(fileKey)
  dataFiles, nFiles = get_files_names( fileKey, inDir, type='cholla' )

  print ('N Files: ', nFiles)

  # fields = ['density']
  if fileKey == 'grid': fields = fields_grid
  if fileKey == 'particles': fields = fields_particles


  stats = None
  for nSnap in range(nFiles):
    if fileKey == 'particles': data_cholla = load_snapshot_data_particles( nSnap, inDir )
    if fileKey == 'grid': data_cholla = load_snapshot_data_grid( nSnap, inDir ) 
    if stats == None:
      stats = {}
      for field in fields:
        stats[field] = {}
        stats[field]['min_vals'] = []
        stats[field]['max_vals'] = []
    for field in fields:
      data = data_cholla[field][...]
      stats[field]['min_vals'].append( data.min() )
      stats[field]['max_vals'].append( data.max() )
  for field in fields:
    stats[field]['min_vals'] = np.array( stats[field]['min_vals'] )
    stats[field]['max_vals'] = np.array( stats[field]['max_vals'] )
    stats[field]['min_global'] = stats[field]['min_vals'].min()
    stats[field]['max_global'] = stats[field]['max_vals'].max()


  outFile = h5.File( outDir + outFileName, 'w' )
  for field in fields:
    group = outFile.create_group( field )
    group.attrs['min_global'] = stats[field]['min_global']
    group.attrs['max_global'] = stats[field]['max_global']
    group.create_dataset( 'min_vals', data =stats[field]['min_vals'] )
    group.create_dataset( 'max_vals', data =stats[field]['max_vals'] )


  outFile.close()
  # 
  # 
  # 
  # 
  # 