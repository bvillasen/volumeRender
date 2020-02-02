import numpy as np
import matplotlib.colors as cl
import matplotlib.cm as cm


colorMaps_matplotlib = [ 'inferno', 'plasma', 'magma', 'viridis', 'jet', 'nipy_spectral', 'CMRmap', 'bone', 'hot', 'copper']
colorMaps_cmocean = [ 'deep_r',  'dense_r', 'haline', 'matter_r', 'thermal']
colorMaps_scientific = [ 'davos', 'devon', 'imola', 'lapaz', 'nuuk', 'oslo']

availble_colormaps = { 'matplotlib':{}, 'palettable':{}}

for c in colorMaps_matplotlib:
  availble_colormaps['matplotlib'][c] = {}
  availble_colormaps['matplotlib'][c]['color_type'] = None
  
for c in colorMaps_cmocean:
  availble_colormaps['palettable'][c] = {}
  availble_colormaps['palettable'][c]['color_type'] = 'cmocean'
  
for c in colorMaps_scientific:
  availble_colormaps['palettable'][c] = {}
  availble_colormaps['palettable'][c]['color_type'] = 'scientific'
    

def get_color_data_from_colormap( colorMap_type, colorMap, nSamples, color_type=None ):
  
  colorVals = np.linspace(0,1,nSamples)

  if colorMap_type == 'matplotlib':
    norm = cl.Normalize(vmin=0, vmax=1, clip=False)
    cmap = cm.ScalarMappable( norm=norm, cmap=colorMap)
    colorData = cmap.to_rgba(colorVals).astype(np.float32)

  if colorMap_type == 'palettable':
    if color_type == 'cmocean':
      import palettable.cmocean.sequential as colors
    if color_type == 'scientific':
      import palettable.scientific.sequential as colors
    
    if colorMap == 'deep_r':  colorMap = colors.Deep_20_r
    if colorMap == 'dense_r':  colorMap = colors.Dense_20_r
    if colorMap == 'haline':  colorMap = colors.Haline_20
    if colorMap == 'matter_r':  colorMap = colors.Matter_20_r
    if colorMap == 'thermal':  colorMap = colors.Thermal_20
    
    
    if colorMap == 'davos': colorMap = colors.Davos_20
    if colorMap == 'devon': colorMap = colors.Devon_20
    if colorMap == 'imola': colorMap = colors.Imola_20
    if colorMap == 'lapaz': colorMap = colors.LaPaz_20
    if colorMap == 'nuuk': colorMap = colors.Nuuk_20
    if colorMap == 'oslo': colorMap = colors.Oslo_20
    
    colorData = colorMap.mpl_colormap( colorVals ).astype(np.float32)

  return colorData



