import os, sys
import numpy as np
import h5py as h5


#Load Snapshot Data

# dataDir = '/raid/bruno/data/'
dataDir = '/data/groups/comp-astro/bruno/'
inDir = dataDir + 'cosmo_sims/1024_hydro_50Mpc/output_files_pchw18/'


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
density = data[data_type]['density']  

