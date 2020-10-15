import os, sys
import h5py as h5
import numpy as np



def get_domain_block( proc_grid, box_size, grid_size ):
  np_x, np_y, np_z = proc_grid
  Lx, Ly, Lz = box_size
  nx_g, ny_g, nz_g = grid_size
  dx, dy, dz = Lx/np_x, Ly/np_y, Lz/np_z
  nx_l, ny_l, nz_l = nx_g//np_x, ny_g//np_y, nz_g//np_z,

  nprocs = np_x * np_y * np_z
  domain = {}
  domain['global'] = {}
  domain['global']['dx'] = dx
  domain['global']['dy'] = dy
  domain['global']['dz'] = dz
  for k in range(np_z):
    for j in range(np_y):
      for i in range(np_x):
        pId = i + j*np_x + k*np_x*np_y
        domain[pId] = { 'box':{}, 'grid':{} }
        xMin, xMax = i*dx, (i+1)*dx
        yMin, yMax = j*dy, (j+1)*dy
        zMin, zMax = k*dz, (k+1)*dz
        domain[pId]['box']['x'] = [xMin, xMax]
        domain[pId]['box']['y'] = [yMin, yMax]
        domain[pId]['box']['z'] = [zMin, zMax]
        domain[pId]['box']['dx'] = dx
        domain[pId]['box']['dy'] = dy
        domain[pId]['box']['dz'] = dz
        domain[pId]['box']['center_x'] = ( xMin + xMax )/2.
        domain[pId]['box']['center_y'] = ( yMin + yMax )/2.
        domain[pId]['box']['center_z'] = ( zMin + zMax )/2.
        gxMin, gxMax = i*nx_l, (i+1)*nx_l
        gyMin, gyMax = j*ny_l, (j+1)*ny_l
        gzMin, gzMax = k*nz_l, (k+1)*nz_l
        domain[pId]['grid']['x'] = [gxMin, gxMax]
        domain[pId]['grid']['y'] = [gyMin, gyMax]
        domain[pId]['grid']['z'] = [gzMin, gzMax]
  return domain
  

def select_procid( proc_id, subgrid, domain, ids, ax ):
  domain_l, domain_r = domain
  subgrid_l, subgrid_r = subgrid
  if domain_l <= subgrid_l and domain_r > subgrid_l:
    ids.append(proc_id)
  if domain_l >= subgrid_l and domain_r <= subgrid_r:
    ids.append(proc_id)
  if domain_l < subgrid_r and domain_r >= subgrid_r:
    ids.append(proc_id)




def select_ids_to_load( subgrid, domain, proc_grid ):
  subgrid_x, subgrid_y, subgrid_z = subgrid
  nprocs = proc_grid[0] * proc_grid[1] * proc_grid[2]
  ids_x, ids_y, ids_z = [], [], []
  for proc_id in range(nprocs):
    domain_local = domain[proc_id]
    domain_x = domain_local['grid']['x']
    domain_y = domain_local['grid']['y']
    domain_z = domain_local['grid']['z']
    select_procid( proc_id, subgrid_x, domain_x, ids_x, 'x' )
    select_procid( proc_id, subgrid_y, domain_y, ids_y, 'y' )
    select_procid( proc_id, subgrid_z, domain_z, ids_z, 'z' )
  set_x = set(ids_x)
  set_y = set(ids_y)
  set_z = set(ids_z)
  set_ids = (set_x.intersection(set_y)).intersection(set_z )
  return list(set_ids)


def load_snapshot_data_distributed( nSnap, inDir, data_type, fields, subgrid,  precision, proc_grid,  box_size, grid_size, show_progess=True ):
  
  
  # Get the doamin domain_decomposition
  domain = get_domain_block( proc_grid, box_size, grid_size )
  
  # Find the ids to load 
  ids_to_load = select_ids_to_load( subgrid, domain, proc_grid )

  print(("Loading Snapshot: {0}".format(nSnap)))
  #Find the boundaries of the volume to load
  domains = { 'x':{'l':[], 'r':[]}, 'y':{'l':[], 'r':[]}, 'z':{'l':[], 'r':[]}, }
  for id in ids_to_load:
    for ax in list(domains.keys()):
      d_l, d_r = domain[id]['grid'][ax]
      domains[ax]['l'].append(d_l)
      domains[ax]['r'].append(d_r)
  boundaries = {}
  for ax in list(domains.keys()):
    boundaries[ax] = [ min(domains[ax]['l']),  max(domains[ax]['r']) ]

  # Get the size of the volume to load
  nx = int(boundaries['x'][1] - boundaries['x'][0])    
  ny = int(boundaries['y'][1] - boundaries['y'][0])    
  nz = int(boundaries['z'][1] - boundaries['z'][0])    

  dims_all = [ nx, ny, nz ]
  data_out = {}
  data_out[data_type] = {}
  for field in fields:
    data_particels = False
    if field in ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']: data_particels = True 
    if not data_particels: data_all = np.zeros( dims_all, dtype=precision )
    else: data_all = []
    added_header = False
    n_to_load = len(ids_to_load)
    for i, nBox in enumerate(ids_to_load):
      name_base = 'h5'
      if data_type == 'particles': inFileName = '{0}_particles.{1}.{2}'.format(nSnap, name_base, nBox)
      if data_type == 'hydro': inFileName = '{0}.{1}.{2}'.format(nSnap, name_base, nBox)
    
      inFile = h5.File( inDir + inFileName, 'r')
      available_fields = inFile.keys()
      head = inFile.attrs
      if added_header == False:
        print( ' Loading: ' + inDir + inFileName )
        print( f' Available Fields:  {available_fields}')
        for h_key in list(head.keys()):
          if h_key in ['dims', 'dims_local', 'offset', 'bounds', 'domain', 'dx', ]: continue
          data_out[h_key] = head[h_key][0]
          if h_key == 'current_z': print((' current_z: {0}'.format( data_out[h_key]) ))
        added_header = True
    
      if show_progess:
        terminalString  = '\r Loading File: {0}/{1}   {2}'.format(i, n_to_load, field)
        sys.stdout. write(terminalString)
        sys.stdout.flush() 
    
      if not data_particels:
        procStart_x, procStart_y, procStart_z = head['offset']
        procEnd_x, procEnd_y, procEnd_z = head['offset'] + head['dims_local']
        # Substract the offsets
        procStart_x -= boundaries['x'][0]
        procEnd_x   -= boundaries['x'][0]
        procStart_y -= boundaries['y'][0]
        procEnd_y   -= boundaries['y'][0]
        procStart_z -= boundaries['z'][0]
        procEnd_z   -= boundaries['z'][0]
        procStart_x, procEnd_x = int(procStart_x), int(procEnd_x)
        procStart_y, procEnd_y = int(procStart_y), int(procEnd_y)
        procStart_z, procEnd_z = int(procStart_z), int(procEnd_z)
        data_local = inFile[field][...]
        data_all[ procStart_x:procEnd_x, procStart_y:procEnd_y, procStart_z:procEnd_z] = data_local
      
      else:
        data_local = inFile[field][...]
        data_all.append( data_local )
    
    if not data_particels:
      # Trim off the excess data on the boundaries:
      trim_x_l = subgrid[0][0] - boundaries['x'][0]
      trim_x_r = boundaries['x'][1] - subgrid[0][1]  
      trim_y_l = subgrid[1][0] - boundaries['y'][0]
      trim_y_r = boundaries['y'][1] - subgrid[1][1]  
      trim_z_l = subgrid[2][0] - boundaries['z'][0]
      trim_z_r = boundaries['z'][1] - subgrid[2][1]  
      trim_x_l, trim_x_r = int(trim_x_l), int(trim_x_r) 
      trim_y_l, trim_y_r = int(trim_y_l), int(trim_y_r) 
      trim_z_l, trim_z_r = int(trim_z_l), int(trim_z_r) 
      data_output = data_all[trim_x_l:nx-trim_x_r, trim_y_l:ny-trim_y_r, trim_z_l:nz-trim_z_r,  ]
      data_out[data_type][field] = data_output
    else:
      data_all = np.concatenate( data_all )
      data_out[data_type][field] = data_all
    if show_progess: print("")
  return data_out



