import os, sys
from os import listdir
from os.path import isfile, join
import h5py as h5
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm


def load_snapshot_data_grid( nSnap, inputDir ):
  inFileName = 'grid_{0}.h5'.format(nSnap)
  snapFile = h5.File( inputDir + inFileName, 'r')
  t = snapFile.attrs['t']
  inputKeys = snapFile.keys()
  # grid_keys = [ 'density', 'momentum_x', 'momentum_y', 'momentum_z', 'Energy']
  # optional_keys = [ 'GasEnergy', 'gravity_density', 'potential', 'potential_grav']
  grid_keys = snapFile.keys()
  data_grid = {}
  data_grid['t'] = t
  # for key in optional_keys:
  #   if key in inputKeys: grid_keys.append( key )
  for key in grid_keys:
    data_grid[key] = snapFile[key]
  return data_grid

# dataDir = '/home/bruno/Desktop/data/'
# inputDir = dataDir + 'cholla_hydro/collapse_3D/'
# nSnap = 0
def load_snapshot_data_particles( nSnap, inputDir ):
  inFileName = 'particles_{0}.h5'.format(nSnap)
  partsFile = h5.File( inputDir + inFileName, 'r')
  fields_data = partsFile.keys()
  current_a = partsFile.attrs['current_a']
  current_z = partsFile.attrs['current_z']
  # particle_mass = partsFile.attrs['particle_mass']

  data_part = {}
  data_part['current_a'] = current_a
  data_part['current_z'] = current_z
  # data_part['particle_mass'] = particle_mass
  part_keys = [ 'density', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z' ]
  extra_keys = [ 'grav_potential', 'mass' ]
  for key in extra_keys:
    if key not in fields_data: continue
    if key in partsFile.keys(): part_keys.append(key)
  for key in part_keys:
    if key not in fields_data: continue
    data_part[key] = partsFile[key]
  return data_part




def load_snapshot_data( nSnap, inDir, cool=False, dm=True, cosmo=True ):
  gridFileName = inDir + 'grid_{0}.h5'.format(nSnap)
  partFileName = inDir + 'particles_{0}.h5'.format(nSnap)
  outDir = {'dm':{}, 'gas':{} }
  data_grid = h5.File( gridFileName, 'r' )
  fields_data = data_grid.keys()
  # print fields_data
  # t = data_grid.attrs['t']
  # dt = data_grid.attrs['dt']
  # outDir['t'] = t
  # outDir['dt'] = dt
  for key in data_grid.attrs.keys(): outDir[key] = data_grid.attrs[key]
  # fields_grid = [ 'density',  'momentum_x', 'momentum_y', 'momentum_z', 'Energy', 'GasEnergy', 'potential', 'extra_scalar', 'extra_scalar_1', 'cooling_rate']
  # if cool: fields_grid.extend(['HI_density', 'HII_density', 'HeI_density', 'HeII_density', 'HeIII_density', 'e_density', 'metal_density', 'temperature', 'flags_DE'])
  fields_grid = fields_data
  for field in fields_grid:
    if field not in fields_data: continue
    outDir['gas'][field] = data_grid[field]

  data_part = h5.File( partFileName, 'r' )
  fields_data = data_part.keys()
  fields_part = [ 'density',  'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z' ]
  # current_z = data_part.attrs['current_z']
  # current_a = data_part.attrs['current_a']
  # outDir['current_a'] = current_a
  # outDir['current_z'] = current_z
  for key in data_part.attrs.keys(): outDir[key] = data_part.attrs[key]
  if cosmo:
    current_z = data_part.attrs['current_z']
    print ("Loading Cholla Snapshot: {0}       current_z: {1}".format( nSnap, current_z) )
  for field in fields_part:
    if field not in fields_data: continue
    # print field
    outDir['dm'][field] = data_part[field]

  return outDir


#
# def load_snapshot_data( nSnap, gridFileName, partFileName ):
#   snapKey = str(nSnap)
#   print "Loading Snapshot: ", nSnap
#   type_all = []
#   if gridFileName != None :
#     # print " Gas"
#     type_all.append('grid')
#     file_grid = h5.File( gridFileName, 'r' )
#     nSnapshots = len(file_grid.keys())
#     data_grid = file_grid[snapKey]
#     # print data_grid.keys()
#     fields_grid = [ 'density',  'momentum_x', 'momentum_y', 'momentum_z', 'Energy']
#     if 'GasEnergy' in data_grid.keys(): fields_grid.append( 'GasEnergy')
#     if 'potential' in data_grid.keys(): fields_grid.append( 'potential')
#
#   if partFileName != None :
#     type_all.append('dm')
#     file_part = h5.File( partFileName, 'r' )
#     nSnapshots = len(file_part.keys())
#     data_part = file_part[snapKey]
#     current_z = data_part.attrs['current_z']
#     current_a = data_part.attrs['current_a']
#
#     fields_dm = [ 'density',  'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z' ]
#
#
#   data_all = {}
#   for data_type in type_all:
#     # print 'Loading ', data_type
#     data_all[data_type] = {}
#
#
#     if data_type == 'grid':
#       data = data_grid
#       fields_out = fields_grid
#       data_all['grid']['t'] = data_grid['t'][0]
#     if data_type == 'dm':
#       data = data_part
#       fields_out = fields_dm
#       data_all['dm']['current_z'] = current_z
#       data_all['dm']['current_a'] = current_a
#     # print ' Fields Available: ', data.keys()
#     for field_out in fields_out:
#       data_field = data[field_out]
#       # print dens.mean()
#       data_all[data_type][field_out] = data_field
#   # file_grid.close()
#   # file_part.close()
#   return data_all, nSnapshots
# #
#
#
#
def change_data( data ):
  data_min = data.min()
  data -= data_min
  # data_new = data
  data_new = np.log10( data + 1)
  data_max = data_new.max()
  return data_new, data_min, data_max


def convert_data(nSnapshots, gridFileName, partFileName ):

  file_grid = h5.File( gridFileName, 'r' )
  file_part = h5.File( partFileName, 'r' )

  outFileName = outDir + 'data_log10_1.h5'
  outFile = h5.File( outFileName, 'w' )

  snapshots = range(nSnapshots)
  for nSnap in snapshots:
    print 'Snapshot: ', nSnap
    snapKey = str(nSnap)
    outSnap = outFile.create_group( snapKey )
    current_a = file_part[snapKey].attrs['current_a']
    current_z = file_part[snapKey].attrs['current_z']
    outSnap.attrs['current_a'] = current_a
    outSnap.attrs['current_z'] = current_z
    print current_z, current_a

    part_types = ['gas', 'dm']
    grid_keys = ['density' ]
    part_keys = ['density']
    for part_type in part_types:
      print ' {0}'.format( part_type )
      file_data = file_grid if part_type == 'gas' else file_part
      outPart = outSnap.create_group( part_type )
      keys = grid_keys if part_type == 'gas' else part_keys
      for key in keys:
        print '  {0}'.format( key)
        data = file_data[snapKey][key][...]
        print data.mean()
        data_new, data_min, data_max = change_data( data )
        outPart.create_dataset( key, data=data_new.astype(np.float32) )
        outPart.attrs['min_'+key] = data_min
        outPart.attrs['max_'+key] = data_max

  for part_type in part_types:
    outPart = outFile.create_group( part_type )
    keys = grid_keys if part_type == 'gas' else part_keys
    for key in keys:
      minKey, maxKey = 'min_'+key, 'max_'+key
      max_all = np.array([ outFile[str(nSnap)][part_type].attrs[maxKey] for nSnap in range(nSnapshots) ])
      min_all = np.array([ outFile[str(nSnap)][part_type].attrs[minKey] for nSnap in range(nSnapshots) ])
      outPart.create_dataset( maxKey, data=max_all )
      outPart.create_dataset( minKey, data=min_all )


  outFile.close()
#
# dataDir = '/home/bruno/Desktop/data/'
#
# inDir = dataDir + 'cosmo_sims/cholla_pm/cosmo_512_hydro/'
# outDir = dataDir + 'cosmo_sims/cholla_pm/cosmo_512_hydro/'
#
# gridFileName = inDir + 'data_grid.h5'
# partFileName = inDir + 'data_particles.h5'
#
#
# nSnapshots = 102
# # convert_data( nSnapshots, gridFileName, partFileName )
#
#
#
#
#
#
# outFileName = outDir + 'data_log10.h5'
# inFile = h5.File( outFileName, 'r' )
#
#
#
#

















# dataDir = '/raid/bruno/data/'
# partFileName = dataDir + '/cosmo_sims/cholla_pm/cosmo_256_hydro/data_particles.h5'
# gridFileName = dataDir + '/cosmo_sims/cholla_pm/cosmo_256_hydro/data_grid.h5'
#
# file_grid = h5.File( gridFileName, 'r' )
