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

# dataDir = '/raid/bruno/data/'
dataDir = '/data/groups/comp-astro/bruno/'
inDir = dataDir + 'cosmo_sims/1024_hydro_50Mpc/output_files_pchw18/'
outDir = dataDir + 'cosmo_sims/1024_hydro_50Mpc/snapshots_prepared/'
create_directory( outDir )

n_snapshot = 169

data_type = 'hydro'
# data_type = 'particles'

fields = ['density']

precision = np.float32

Lbox = 5000    #kpc/h
proc_grid = [ 4, 2, 2]
box_size = [ Lbox, Lbox, Lbox ]
grid_size = [ 1024, 1024, 1024 ] #Size of the simulation grid
subgrid = [ [0, 1024], [0, 1024], [0, 1024] ] #Size of the volume to load
data = load_snapshot_data_distributed( n_snapshot, inDir, data_type, fields, subgrid,  precision, proc_grid,  box_size, grid_size, show_progess=True )

field = 'density'

data_vals = data[data_type][field]  


# Normalize Data
max_val = data_vals.max() / 10
min_val = data_vals.min()
data_vals = np.clip( data_vals, a_min=min_val, a_max=max_val ) 
data_vals = ( data_vals - min_val ) / ( max_val - min_val )

# Change to 256 range
data_vals = (255*(data_vals)).astype(np.uint8)

#Write to file
out_file_name = outDir + 'snapshot_{0}.h5'.format( n_snapshot )
out_file = h5.File( out_file_name, 'w')

out_file.close()
print( "Saved File: " + out_file_name )
