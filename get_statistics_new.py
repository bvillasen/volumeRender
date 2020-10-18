import os, sys
import numpy as np
import h5py as h5

#Add Modules from other directories
currentDirectory = os.getcwd()
dataDirectory = currentDirectory + "/data_src/"
sys.path.extend([  dataDirectory ] )
from tools import *
from load_data_cholla import load_snapshot_data_particles, load_snapshot_data_grid

nPoints = 1024

# dataDir = '/raid/bruno/data/'
dataDir = '/data/groups/comp-astro/bruno/'
inDir = dataDir + 'cosmo_sims/{0}_hydro_50Mpc/output_files_pchw18/'.format(nPoints)
outDir = dataDir + 'cosmo_sims/{0}_hydro_50Mpc/output_files_pchw18/statistics/'.format(nPoints)
create_directory( outDir )



data_type = 'hydro'
fields = ['density' ]



nFiles = 3

stats = None
for nSnap in range(nFiles):
  
  precision = np.float32
  Lbox = 5000    #kpc/h
  if nPoints == 1024: proc_grid = [ 4, 2, 2]
  if nPoints == 2048: proc_grid = [ 8, 8, 8]
  box_size = [ Lbox, Lbox, Lbox ]
  grid_size = [ nPoints, nPoints, nPoints ] #Size of the simulation grid
  subgrid = [ [0, nPoints], [0, nPoints], [0, nPoints] ] #Size of the volume to load
  data = load_snapshot_data_distributed( n_snapshot, inDir, data_type, fields, subgrid,  precision, proc_grid,  box_size, grid_size, show_progess=True, get_statistics=True )
  
  if stats == None:
    stats = {}
    for field in fields:
      stats[field] = {}
      stats[field]['min_vals'] = []
      stats[field]['max_vals'] = []
  for field in fields:
    print data_cholla.keys()
    # data = data_cholla[field][...]
    # stats[field]['min_vals'].append( data.min() )
    # stats[field]['max_vals'].append( data.max() )
    # data = data_cholla[field]
    # stats[field]['min_vals'].append( data.attrs['min'] )
    # stats[field]['max_vals'].append( data.attrs['max'] )
    stats[field]['min_vals'].append( data[data_type]['statistics'][field]['min'] )
    stats[field]['max_vals'].append( data[data_type]['statistics'][field]['max']  )

# print( "nSnapshot {0}:  {1} {2}".format( nSnapshot, stats[field]['min_vals'], stats[field]['max_vals']  )
for field in fields:
  stats[field]['min_vals'] = np.array( stats[field]['min_vals'] )
  stats[field]['max_vals'] = np.array( stats[field]['max_vals'] )
  stats[field]['min_global'] = stats[field]['min_vals'].min()
  stats[field]['max_global'] = stats[field]['max_vals'].max()
  print '{0}: min:{1}'.format( fields, stats[field]['min_vals']  )
  print '{0}: max:{1}'.format( fields, stats[field]['max_vals']  )


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
