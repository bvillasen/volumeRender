import os, sys
import numpy as np
import h5py as h5

currentDirectory = os.getcwd()
srcDirectory = currentDirectory + "/src/"
dataDirectory = currentDirectory + "/data_src/"
sys.path.extend([ srcDirectory, dataDirectory ] )
from tools import create_directory
from load_data_cholla_distributed import load_snapshot_data_distributed

#Load Snapshot Data
nPoints = 1024

# dataDir = '/raid/bruno/data/'
dataDir = '/data/groups/comp-astro/bruno/'
inDir = dataDir + 'cosmo_sims/{0}_hydro_50Mpc/output_files_pchw18/'.format(nPoints)
outDir = dataDir + 'cosmo_sims/{0}_hydro_50Mpc/snapshots_prepared/'.format(nPoints)
create_directory( outDir )


n_snapshot = 169
for n_snapshot in range(170):

  data_type = 'hydro'
  # data_type = 'particles'

  fields = ['density']

  precision = np.float32

  Lbox = 5000    #kpc/h
  if nPoints == 1024: proc_grid = [ 4, 2, 2]
  if nPoints == 2048: proc_grid = [ 8, 8, 8]
  box_size = [ Lbox, Lbox, Lbox ]
  grid_size = [ nPoints, nPoints, nPoints ] #Size of the simulation grid
  subgrid = [ [0, nPoints], [0, nPoints], [0, nPoints] ] #Size of the volume to load
  data = load_snapshot_data_distributed( n_snapshot, inDir, data_type, fields, subgrid,  precision, proc_grid,  box_size, grid_size, show_progess=True )

  field = 'density'

  min_val = 0.020682
  max_val = 6375.8875

  data_vals = data[data_type][field]  
  # min_val = data_vals.min()
  data_vals -= min_val


  # Normalize Data
  # max_val = data_vals.max()
  max_val = max_val / 1000 
  data_vals = np.clip( data_vals, a_min=None, a_max=max_val ) 
  data_vals = np.log10(data_vals + 1) / np.log10( max_val + 1)


  # print( "Min: {0}   Max: {1}".format(min_val, max_val ))

  # Change to 256 range
  data_vals = (255*(data_vals)).astype(np.uint8)

  #Write to file
  out_file_name = outDir + '{0}_{1}_{2}.h5'.format( data_type, field, n_snapshot )
  out_file = h5.File( out_file_name, 'w')
  out_file.create_dataset( field, data=data_vals )
  out_file.close()
  print( "Saved File: " + out_file_name )
