import sys, time, os
from os import listdir
from os.path import isfile, join
import h5py as h5
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import matplotlib.colors as cl
import matplotlib.cm as cm

#Add Modules from other directories
currentDirectory = os.getcwd()
srcDirectory = currentDirectory + "/src/"
dataDirectory = currentDirectory + "/data_src/"
sys.path.extend([ srcDirectory, dataDirectory ] )
import volumeRender_single as volumeRender
from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, np3DtoCudaArray
from render_functions import *
from data_functions import *

# dataDir = '/home/bruno/Desktop/data/'
# dataDir = '/home/bruno/Desktop/hdd_extrn_1/data/'
dataDir = '/home/bruno/Desktop/ssd_0/data/'
# inDir = dataDir + 'summit/1024_cool_uv_50Mpc/output_snapshots/'
# inDir = dataDir + 'cosmo_sims/cholla_pm/256_cool_uv_50Mpc/data_PPMC_HLLC_SIMPLE_eta0.001_0.0400/'
# inDir = dataDir + 'cosmo_sims/cholla_pm/sphere_explosion/data_ppmp/'
# inDir = dataDir + 'cosmo_sims/cholla_pm/256_dm_50Mpc/data/'
inDir = dataDir + 'cosmo_sims/1024_hydro_50Mpc/snapshots_prepared/'
outDir = '/home/bruno/Desktop/cosmo_gas/'


#Select CUDA Device
useDevice = 0

nSnap = 169

nFields = 1

n_image = 0


load_stats = False

rotation_angle = 60



data_format = 'cholla'
data_type = 'particles'
data_field = 'density'
# normalization = 'global'
normalization = 'local'
log_data = True
n_border = 2

if normalization == 'global': load_stats = True

data_parameters_default = { 'data_format': data_format, 'normalization':normalization, 'log_data':log_data, 'n_border':n_border }

data_parameters = {}

# data_parameters[0] = data_parameters_default.copy()
# data_parameters[0]['data_type'] = 'particles'
# data_parameters[0]['data_field'] = 'density'

data_parameters[0] = data_parameters_default.copy()
data_parameters[0]['data_type'] = 'hydro'
data_parameters[0]['data_field'] = 'density'

# data_parameters[0] = data_parameters_default.copy()
# data_parameters[0]['data_type'] = 'hydro'
# data_parameters[0]['data_field'] = 'temperature'

volumeRender.render_text['x'] = -0.45
volumeRender.render_text['y'] = 0.45
volumeRender.render_text['text'] = ''




data_prepared = True
data_to_render_list = [ get_Data_to_Render( nSnap, inDir, data_parameters[i], stats=load_stats, data_prepared=data_prepared ) for i in range(nFields)]


#Get Dimensions of the data to render
nz, ny, nx = data_to_render_list[0].shape
nWidth, nHeight, nDepth = nx, ny, nz

#Set the parameters for rendering each field
volumeRender.render_parameters[0] = { 'transp_type':'sigmoid', 'colormap':{}, 'transp_center':0., "transp_ramp": 2., 'density':0.03, "brightness":2.0, 'transfer_offset': 0.05, 'transfer_scale': 1.2 }
volumeRender.render_parameters[0]['colormap']['main'] = 'palettable'
volumeRender.render_parameters[0]['colormap']['name'] = 'haline'
volumeRender.render_parameters[0]['colormap']['type'] = 'cmocean'


# # DM Density
# volumeRender.render_parameters[0] = { 'transp_type':'sigmoid', 'colormap':{}, 'transp_center':0.46, "transp_ramp": 2.0, 'density':0.089, "brightness":2.0, 'transfer_offset': 0, 'transfer_scale': 1.0 }
# volumeRender.render_parameters[0]['colormap']['main'] = 'matplotlib'
# volumeRender.render_parameters[0]['colormap']['name'] = 'CMRmap'
# 
# # Gas Density
# volumeRender.render_parameters[0] = { 'transp_type':'sigmoid', 'colormap':{}, 'transp_center':-0.13, "transp_ramp": 2.25, 'density':0.03, "brightness":2.0, 'transfer_offset': 0.049, 'transfer_scale': 1.88 }
# volumeRender.render_parameters[0]['colormap']['main'] = 'palettable'
# volumeRender.render_parameters[0]['colormap']['name'] = 'haline'
# volumeRender.render_parameters[0]['colormap']['type'] = 'cmocean'


# # Gas Temperature
# volumeRender.render_parameters[0] = { 'transp_type':'sigmoid',  'colormap':{}, 'transp_center':0.3, "transp_ramp": 3., 'density':0.01, "brightness":2.0, 'transfer_offset':0.06 , 'transfer_scale': 1.15 }
# volumeRender.render_parameters[0]['colormap']['main'] = 'matplotlib'
# volumeRender.render_parameters[0]['colormap']['name'] = 'jet'

#Initialize openGL
volumeRender.width_GL = int( 512*4  )
volumeRender.height_GL = int( 512*4  )
volumeRender.nTextures = nFields
volumeRender.nWidth = nWidth
volumeRender.nHeight = nHeight
volumeRender.nDepth = nDepth
volumeRender.windowTitle = "Cosmo Volume Render"
volumeRender.initGL()

#set thread grid for CUDA kernels
grid3D, block3D = volumeRender.get_CUDA_threads( 8, 8, 8)   #hardcoded, tune to your needs

#initialize pyCUDA context
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=True )


#Initialize all gpu data
copyToScreen_list = volumeRender.Initialize_GPU_Data( nFields, data_to_render_list ) 

########################################################################
send_data = True
def sendToScreen( ):
  global send_data
  if send_data:
    for i in range(nFields): 
      copyToScreen_list[i]
    send_data = False 
########################################################################

def stepFunction():
  global  nSnap 
  # volumeRender.render_parameters[0]['transp_center'] = volumeRender.set_transparency_center( nSnap, z)
  # print "Transparency center = {0}".format(volumeRender.render_parameters[0]['transp_center'])
  # volumeRender.Change_Rotation_Angle( rotation_angle )
  sendToScreen( )

########################################################################
def specialKeyboardFunc( key, x, y ):
  global nSnap
  if key== volumeRender.GLUT_KEY_RIGHT:
    volumeRender.color_second_index += 1
    volumeRender.changed_colormap = True
  if key== volumeRender.GLUT_KEY_LEFT:
    volumeRender.color_second_index -= 1
    volumeRender.changed_colormap = True  
  if key== volumeRender.GLUT_KEY_UP:
    volumeRender.color_first_index += 1
    volumeRender.changed_colormap = True
  if key== volumeRender.GLUT_KEY_DOWN:
    volumeRender.color_first_index -= 1
    volumeRender.changed_colormap = True

  # if key== volumeRender.GLUT_KEY_RIGHT:
  #   nSnap += 1
  #   if nSnap == nSnapshots: nSnap = 0
  #   print " Snapshot: ", nSnap
  #   change_snapshot( nSnap )

def keyboard(*args):
  ESCAPE = '\033'
  SPACE = '32'
  # If escape is pressed, kill everything.
  if args[0] == ESCAPE:
    print "Ending Simulation"
    #cuda.gl.Context.pop()
    sys.exit()  
  if args[0] == 'z':
      print "Saving Image: {0}".format( volumeRender.n_image)
      volumeRender.save_image(dir=outDir, image_name='image')

  if args[0] == 'q':
    volumeRender.render_parameters[0]['transp_center'] -= np.float32(0.01)
    print "Image Transp Center: ",volumeRender.render_parameters[0]['transp_center']
  if args[0] == 'w':
    volumeRender.render_parameters[0]['transp_center'] += np.float32(0.01)
    print "Image Transp Center: ",volumeRender.render_parameters[0]['transp_center']
  if args[0] == 'a':
    volumeRender.render_parameters[0]['transp_ramp'] -= np.float32(0.01)
    print "Image Transp Ramp: ",volumeRender.render_parameters[0]['transp_ramp']
  if args[0] == 's':
    volumeRender.render_parameters[0]['transp_ramp'] += np.float32(0.01)
    print "Image Transp Ramp: ",volumeRender.render_parameters[0]['transp_ramp']
  if args[0] == 'd':
    volumeRender.render_parameters[0]['density'] -= np.float32(0.01)
    print "Image Density: ",volumeRender.render_parameters[0]['density']
  if args[0] == 'e':
    volumeRender.render_parameters[0]['density'] += np.float32(0.01)
    print "Image Density: ",volumeRender.render_parameters[0]['density']
  if args[0] == 'f':
    volumeRender.render_parameters[0]['brightness'] -= np.float32(0.01)
    print "Image brightness: ",volumeRender.render_parameters[0]['brightness']
  if args[0] == 'r':
    volumeRender.render_parameters[0]['brightness'] += np.float32(0.01)
    print "Image brightness: ",volumeRender.render_parameters[0]['brightness']
  if args[0] == 't':
    volumeRender.render_parameters[0]['transfer_offset'] -= np.float32(0.01)
    print "Image transfer_offset: ",volumeRender.render_parameters[0]['transfer_offset']
  if args[0] == 'g':
    volumeRender.render_parameters[0]['transfer_offset'] += np.float32(0.01)
    print "Image transfer_offset: ",volumeRender.render_parameters[0]['transfer_offset']
  if args[0] == 'y':
    volumeRender.render_parameters[0]['transfer_scale'] -= np.float32(0.01)
    print "Image transfer_scale: ",volumeRender.render_parameters[0]['transfer_scale']
  if args[0] == 'h':
    volumeRender.render_parameters[0]['transfer_scale'] += np.float32(0.01)
    print "Image transfer_scale: ",volumeRender.render_parameters[0]['transfer_scale']


  # if args[0] == '2':
  #   transferScale -= np.float32(0.01)
  #   print "Image Transfer Scale: ",transferScale
  # if args[0] == '4':
  #   brightness -= np.float32(0.01)

########################################################################
#configure volumeRender functions
volumeRender.specialKeys = specialKeyboardFunc
volumeRender.stepFunc = stepFunction
volumeRender.keyboard = keyboard
#run volumeRender animation
volumeRender.animate()
