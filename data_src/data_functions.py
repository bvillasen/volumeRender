import numpy as np
import h5py as h5
from load_data_cholla import load_snapshot_data_particles, load_snapshot_data_grid



global_data_parameters = {}

parameters_dm_50Mpc = { 'transp_type':'sigmoid', 'cmap_indx':0, 'transp_center':0, "transp_ramp": 2.5, 'density':0.03, "brightness":2.0, 'transfer_offset': 0, 'transfer_scale': 1 }


def get_data( nSnap, inDir, data_parameters, stats=None ):
  global global_parameters
  format = data_parameters['data_format']
  type = data_parameters['data_type']
  field = data_parameters['data_field']
  data_dic = {}
  if format == 'cholla':
    if type == 'particles':
      data_cholla = load_snapshot_data_particles( nSnap, inDir )
      if data_cholla.get('current_z') != None:
        global_data_parameters['current_z'] = data_cholla['current_z']
        global_data_parameters['current_z'] = data_cholla['current_z']
        global_data_parameters['current_a'] = data_cholla['current_a']
        data_dic['current_z'] = data_cholla['current_z']
        data_dic['current_a'] = data_cholla['current_a']
      # data = data_cholla[field][...]
      data_dic['data'] = data
    if type == 'grid':
      data_cholla = load_snapshot_data_grid( nSnap, inDir )
      data = data_cholla[field][...]
      # #Add Global min:
      # min_clip = None
      # max_clip = None
      # print "Applying Clip: min:{0}    max:{1}".format(min_clip, max_clip)
      # data = np.clip( data, a_min = min_clip, a_max=max_clip)
      # data = data_cholla[field][:, :256, :256]
      data_dic['data'] = data
      if data_cholla.get('current_z') != None:
        global_data_parameters['current_z'] = data_cholla['current_z']
        global_data_parameters['current_z'] = data_cholla['current_z']
        global_data_parameters['current_a'] = data_cholla['current_a']
        data_dic['current_z'] = data_cholla['current_z']
        data_dic['current_a'] = data_cholla['current_a']
    if stats:
      stats_file = h5.File( inDir + 'stats_{0}.h5'.format(type), 'r')
      stats_field = stats_file[field]
      min_global = stats_field.attrs['min_global']
      max_global = stats_field.attrs['max_global']
      min_vals = stats_field['min_vals'][...]
      max_vals = stats_field['max_vals'][...]
      stats_file.close()
      data_dic['stats'] = {}
      data_dic['stats']['min_global'] = min_global
      data_dic['stats']['max_global'] = max_global
      data_dic['stats']['min_vals'] = min_vals
      data_dic['stats']['max_vals'] = max_vals
  return data_dic
      

def get_Data_to_Render( nSnap, inDir, data_parameters, stats=True,  ):
  data_dic = get_data( nSnap, inDir, data_parameters, stats=stats )
  data_to_render = data_dic['data']
  if stats: stats_dic = data_dic['stats']
  else: stats_dic = None
  plotData = prepare_data( data_to_render, data_parameters, stats=stats_dic )
  return plotData
  

def get_Data_for_Interpolation( nSnap, inDir, data_parameters, n_snapshots, data_for_interpolation=None,  ):
  if data_for_interpolation == None: 
    data_for_interpolation = {}
    nSnap_0 = nSnap
    nSnap_1 = nSnap_0 + 1
    print(" Lodading Snapshot: {0}").format(nSnap_0)
    data_dic_0 = get_data( nSnap_0, inDir, data_parameters, stats=True )
    data_0 = data_dic_0['data']
    stats = data_dic_0['stats']
    print(" Lodading Snapshot: {0}").format(nSnap_1)
    data_dic_1 = get_data( nSnap_1, inDir, data_parameters, stats=False )
    data_1 = data_dic_1['data']
    data_for_interpolation['stats'] = stats
    data_for_interpolation['nSnap'] = nSnap_0
    data_for_interpolation[0] = data_0
    data_for_interpolation[1] = data_1
    data_for_interpolation['z'] = {}
    data_for_interpolation['z'][0] = data_dic_0['current_z']
    data_for_interpolation['z'][1] = data_dic_1['current_z']
  else:
    nSnap_prev = data_for_interpolation['nSnap']
    if nSnap - nSnap_prev != 1: print( 'ERROR: Interpolation snapshot sequence')
    nSnap_0 = nSnap
    nSnap_1 = nSnap_0 + 1
    data_for_interpolation['nSnap'] = nSnap_0
    print( " Swaping data snapshot: {0}".format(nSnap_0) )
    data_for_interpolation[0] = data_for_interpolation[1].copy()
    if nSnap_1 == n_snapshots:
      print( "Exiting: Interpolation") 
      data_1 = data_for_interpolation[0]
    else:
      print(" Lodading Snapshot: {0}").format(nSnap_1)
      data_dic_1 = get_data( nSnap_1, inDir, data_parameters, stats=False )
      data_1 = data_dic_1['data']
      
      data_for_interpolation['z'][0] = data_for_interpolation['z'][1]
      data_for_interpolation['z'][1] = data_dic_1['current_z']
    data_for_interpolation[1] = data_1
  return data_for_interpolation

def get_Data_List_to_Render_Interpolation( nSnap, inDir, nFields, current_frame, frames_per_snapshot, data_parameters, data_for_interpolation, n_snapshots ):
  if data_for_interpolation == None:
    data_for_interpolation = {}
    for i in range(nFields):
      data_for_interpolation[i] = None

  data_to_render_list = []
  for i in range( nFields ):
    if current_frame % frames_per_snapshot == 0:
      data_for_interpolation[i] = get_Data_for_Interpolation( nSnap, inDir, data_parameters[i], n_snapshots, data_for_interpolation=data_for_interpolation[i]  )
    stats = data_for_interpolation[i]['stats']
    data_interpolated, z_interpolated = Interpolate_Data( current_frame, frames_per_snapshot, data_for_interpolation[i] )
    plotData = prepare_data( data_interpolated, data_parameters[0], stats=stats )
    data_to_render_list.append( plotData )
  return data_to_render_list, data_for_interpolation, z_interpolated



def Interpolate_Data( current_frame, frames_per_snapshot, data_for_interpolation ):
  nSnap = data_for_interpolation['nSnap']
  data_0 = data_for_interpolation[0]
  data_1 = data_for_interpolation[1]
  z_0 = data_for_interpolation['z'][0]
  z_1 = data_for_interpolation['z'][1]
  a_0 = 1. / (z_0 + 1)
  a_1 = 1. / (z_1 + 1)
  alpha = float( current_frame % frames_per_snapshot ) / frames_per_snapshot
  print( ' Interpolating Snapshots  {0} -> {1}   alpha:{2}'.format( nSnap, nSnap+1, alpha) )
  if alpha >= 1: print('ERROR: Interpolation alpha >= 1')
  data_interpolated = data_0 + alpha * ( data_1 - data_0 )
  a_interpolated = a_0 + alpha * ( a_1 - a_0 )
  z_interpolated = 1./a_interpolated - 1
  return data_interpolated, z_interpolated
  
def set_frame( data, n ):
  val = 1.0
  data[:,:n,:n] = val
  data[:n,:,:n] = val
  data[:n,:n,:] = val
  data[-n:,-n:,:] = val
  data[:,-n:,-n:] = val
  data[-n:,:,-n:] = val
  data[-n:,:n,:] = val
  data[-n:,:,:n] = val
  data[:,:n,-n:] = val
  data[:,-n:,:n] = val
  data[:n,:,-n:] = val
  data[:n,-n:,:] = val
  return data


def prepare_data( plotData,  data_parameters, stats=None ):
  
  log = data_parameters['log_data']
  normalize = data_parameters['normalization']
  n_border = data_parameters['n_border']
  
  if normalize == 'local':
    print plotData.max()
    if log : plotData = np.log10(plotData + 1)
    # plotData -= plotData.min()
    norm_val = plotData.max()
    plotData /= norm_val
    
  if normalize == 'global':
    max_global = stats['max_global']
    min_global = stats['min_global']
    print "Global min:{0}    max{1}".format( min_global, max_global)
    max_all = max_global - min_global
    plotData -= min_global
    if log :
      
      log_max =  np.log10( max_all + 1)
      plotData = np.log10(plotData + 1)
      plotData /= log_max
    
  plotData = set_frame(plotData, n_border)
  plotData_h_256 = (255*(plotData)).astype(np.uint8)
  return plotData_h_256