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
  

def Change_Snapshot_Single_Field( nSnap, field_index, copyToScreen_list, inDir, data_parameters, stats=False  ):
  plotData = get_Data_to_Render( nSnap, inDir, data_parameters, stats=stats )
  copyToScreen = copyToScreen_list[field_index]
  copyToScreen.set_src_host(plotData)
  copyToScreen()
  
  
  
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