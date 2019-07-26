import numpy as np
import h5py as h5
from load_data_cholla import load_snapshot_data_particles






def get_data( nSnap, inDir, data_parameters, stats=None ):
  format = data_parameters['data_format']
  type = data_parameters['data_type']
  field = data_parameters['data_field']
  data_dic = {}
  if format == 'cholla':
    if type == 'particles':
      data_cholla = load_snapshot_data_particles( nSnap, inDir )
      data = data_cholla[field][...]
      data_dic['data'] = data
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
  data_dic = get_data( nSnap, inDir, data_parameters, stats=True )
  data_to_render = data_dic['data']
  stats_dic = data_dic['stats']
  plotData = prepare_data( data_to_render, data_parameters, stats=stats_dic )
  return plotData
  

def get_Data_for_Interpolation( nSnap, inDir, data_parameters,  data_for_interpolation=None ):
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
  else:
    nSnap_prev = data_for_interpolation['nSnap']
    if nSnap - nSnap_prev != 1: print( 'ERROR: Interpolation snapshot sequence')
    nSnap_0 = nSnap
    nSnap_1 = nSnap_0 + 1
    data_for_interpolation['nSnap'] = nSnap_0
    print( " Swaping data snapshot: {0}".format(nSnap_0) )
    data_for_interpolation[0] = data_for_interpolation[1].copy()
    print(" Lodading Snapshot: {0}").format(nSnap_1)
    data_dic_1 = get_data( nSnap_1, inDir, data_parameters, stats=False )
    data_1 = data_dic_1['data']
    data_for_interpolation[1] = data_1
  return data_for_interpolation

def get_Data_List_to_Render_Interpolation( nSnap, inDir, nFields, current_frame, frames_per_snapshot, data_parameters, data_for_interpolation ):
  if data_for_interpolation == None:
    data_for_interpolation = {}
    for i in range(nFields):
      data_for_interpolation[i] = None

  data_to_render_list = []
  for i in range( nFields ):
    if current_frame % frames_per_snapshot == 0:
      data_for_interpolation[i] = get_Data_for_Interpolation( nSnap, inDir, data_parameters[i], data_for_interpolation=data_for_interpolation[i]  )
    stats = data_for_interpolation[i]['stats']
    data_interpolated = Interpolate_Data( current_frame, frames_per_snapshot, data_for_interpolation[i] )
    plotData = prepare_data( data_interpolated, data_parameters[0], stats=stats )
    data_to_render_list.append( plotData )
  return data_to_render_list, data_for_interpolation



def Interpolate_Data( current_frame, frames_per_snapshot, data_for_interpolation ):
  nSnap = data_for_interpolation['nSnap']
  data_0 = data_for_interpolation[0]
  data_1 = data_for_interpolation[1]
  alpha = float( current_frame % frames_per_snapshot ) / frames_per_snapshot
  print( ' Interpolating Snapshots  {0} -> {1}   alpha:{2}'.format( nSnap, nSnap+1, alpha) )
  if alpha >= 1: print('ERROR: Interpolation alpha >= 1')
  data_interpolated = data_0 + alpha * ( data_1 - data_0 )
  return data_interpolated
  
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
    if log : plotData = np.log10(plotData + 1)
    plotData -= plotData.min()
    norm_val = plotData.max()
    plotData /= norm_val
    
  if normalize == 'global':
    max_global = stats['max_global']
    min_global = stats['min_global']
    max_all = max_global - min_global
    plotData -= min_global
    if log :
      log_max =  np.log10( max_all + 1)
      plotData = np.log10(plotData + 1)
      plotData /= log_max
    
  plotData = set_frame(plotData, n_border)
  plotData_h_256 = (255*(plotData)).astype(np.uint8)
  return plotData_h_256