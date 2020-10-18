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
import volumeRender_new as volumeRender
from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, np3DtoCudaArray
from render_functions import *
from data_functions import *
from tools import create_directory

# dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
dataDir = '/home/bruno/Desktop/ssd_0/data/'
# inDir = dataDir + 'cosmo_sims/cholla_pm/256_dm_50Mpc/data/'
# inDir = dataDir + 'cosmo_sims/cholla_pm/128_cool/data_float32/'
inDir = dataDir + 'cosmo_sims/1024_hydro_50Mpc/snapshots_pchw18/hydro_temperature/'
# outDir = 'image_output/'
outDir = inDir + 'render_temperature/'
create_directory( outDir )

#Select CUDA Device
useDevice = 0

nFields = 1

interpolation = True
data_for_interpolation = None

save_images = True

n_snapshots = 200
snapshots = range(0, n_snapshots, 1) 

n_frames = n_snapshots*10
frames_per_snapshot = n_frames / n_snapshots

rotation_angle = 0
total_rotation = 180.
delta_rotation = float(total_rotation) / (n_frames-2)


stats = None

# Count the frames to change snapshot
current_frame = 0

data_format = 'cholla'
data_type = 'particles'
data_field = 'density'
normalization = 'global'
log_data = True
n_border = 3

data_parameters_default = { 'data_format': data_format, 'normalization':normalization, 'log_data':log_data, 'n_border':n_border }

data_parameters = {}
# data_parameters[0] = data_parameters_default.copy()
# data_parameters[0]['data_type'] = 'particles'
# data_parameters[0]['data_field'] = 'density'

# data_parameters[0] = data_parameters_default.copy()
# data_parameters[0]['data_type'] = 'grid'
# data_parameters[0]['data_field'] = 'density'

data_parameters[0] = data_parameters_default.copy()
data_parameters[0]['data_type'] = 'grid'
data_parameters[0]['data_field'] = 'temperature'

volumeRender.render_text['x'] = -0.45
volumeRender.render_text['y'] =  0.45



#Initial Snapshot
nSnap = 0

#Restart
# nSnap = 186
# current_frame = nSnap * frames_per_snapshot
# volumeRender.n_image = current_frame - 1
# rotation_angle = (current_frame - 1) * delta_rotation

if interpolation:
  data_to_render_list, data_for_interpolation, current_z = get_Data_List_to_Render_Interpolation( nSnap, inDir, nFields, current_frame, frames_per_snapshot, data_parameters,  data_for_interpolation, n_snapshots )

 
else:
  data_to_render_list = [ get_Data_to_Render( nSnap, inDir, data_parameters[i], stats=True ) for i in range(nFields)]


volumeRender.render_text['x'] = -0.45
volumeRender.render_text['y'] =  0.45
volumeRender.render_text['text'] = 'z = {0:.2f}'.format(current_z)


#Get Dimensions of the data to render
nz, ny, nx = data_to_render_list[0].shape
nWidth, nHeight, nDepth = nx, ny, nz

#Set the parameters for rendering each field
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

# Gas Temperature
volumeRender.render_parameters[0] = { 'transp_type':'sigmoid',  'colormap':{}, 'transp_center':0.3, "transp_ramp": 3, 'density':0.01, "brightness":2.0, 'transfer_offset':0.06 , 'transfer_scale': 1.15 }
volumeRender.render_parameters[0]['colormap']['main'] = 'matplotlib'
volumeRender.render_parameters[0]['colormap']['name'] = 'jet'



#Initialize openGL
volumeRender.width_GL = 512*4
volumeRender.height_GL = 512*4
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
      copyToScreen_list[i]()
    send_data = False 
########################################################################

exit_program = False
def stepFunction():
  global exit_program, nSnap, current_frame, rotation_angle, data_for_interpolation, send_data, data_to_render_list, copyToScreen_list, current_z
  
  sendToScreen( )
  # print nSnap
  if current_frame > 0 and save_images: volumeRender.save_image(dir=outDir, image_name='image')
  if current_frame > 0: rotation_angle += delta_rotation
  volumeRender.Change_Rotation_Angle( rotation_angle )
  if interpolation:
    if exit_program:
      print "Finished Animation" 
      exit()
    current_frame, nSnap = volumeRender.Update_Frame_Number( nSnap, current_frame, frames_per_snapshot )
    if nSnap == n_snapshots-1 and current_frame%frames_per_snapshot == frames_per_snapshot-1:
      print( "Exiting")
      exit_program = True
    data_to_render_list, data_for_interpolation, current_z = get_Data_List_to_Render_Interpolation( nSnap, inDir, nFields, current_frame, frames_per_snapshot, data_parameters, data_for_interpolation, n_snapshots )
    volumeRender.render_text['text'] = 'z = {0:.2f}'.format(current_z)
    volumeRender.render_parameters[0]['transp_center'] = volumeRender.set_transparency_center( nSnap, current_z)
    print "Transparency center = {0}".format(volumeRender.render_parameters[0]['transp_center'])
    volumeRender.Change_Data_to_Render( nFields, data_to_render_list, copyToScreen_list )
  else:
    current_frame += 1
    if current_frame % frames_per_snapshot == 0:
      nSnap += 1
      if nSnap == n_snapshots:
        print "Finished Animation" 
        exit()
      print "Change Snapshot: {0}  Frame:{1}".format( nSnap, current_frame)
      for i in range(nFields):
        volumeRender.Change_Snapshot_Single_Field( nSnap, i, copyToScreen_list, inDir, data_parameters[i], stats=True  )

########################################################################
def specialKeyboardFunc( key, x, y ):
  global nSnap
  if key== volumeRender.GLUT_KEY_RIGHT:
    nSnap += 1
    if nSnap == nSnapshots: nSnap = 0
    print " Snapshot: ", nSnap
    change_snapshot( nSnap )

########################################################################
#configure volumeRender functions
volumeRender.specialKeys = specialKeyboardFunc
volumeRender.stepFunc = stepFunction
# volumeRender.keyboard = keyboard
#run volumeRender animation
volumeRender.animate()
