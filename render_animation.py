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

dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
# inDir = dataDir + 'cosmo_sims/cholla_pm/256_dm_50Mpc/data/'
inDir = dataDir + 'cosmo_sims/cholla_pm/128_cool/data_float32/'
outDir = 'image_output/'

#Select CUDA Device
useDevice = 0

nFields = 3

interpolation = True
data_for_interpolation = None

save_images = True

n_snapshots = 109
snapshots = range(0, n_snapshots, 1) 

n_frames = n_snapshots*1
frames_per_snapshot = n_frames / n_snapshots

rotation_angle = 0
total_rotation = 360
delta_rotation = float(total_rotation) / (n_frames-2)


stats = None

# Count the frames to change snapshot
current_frame = 0

data_format = 'cholla'
data_type = 'particles'
data_field = 'density'
normalization = 'global'
log_data = True
n_border = 1

data_parameters_default = { 'data_format': data_format, 'normalization':normalization, 'log_data':log_data, 'n_border':n_border }

data_parameters = {}
data_parameters[0] = data_parameters_default.copy()
data_parameters[0]['data_type'] = 'particles'
data_parameters[0]['data_field'] = 'density'

data_parameters[1] = data_parameters_default.copy()
data_parameters[1]['data_type'] = 'grid'
data_parameters[1]['data_field'] = 'density'

data_parameters[2] = data_parameters_default.copy()
data_parameters[2]['data_type'] = 'grid'
data_parameters[2]['data_field'] = 'temperature'



#Initial Snapshot
nSnap = 0

if interpolation:
  data_to_render_list, data_for_interpolation = get_Data_List_to_Render_Interpolation( nSnap, inDir, nFields, current_frame, frames_per_snapshot, data_parameters,  data_for_interpolation, n_snapshots )

 
else:
  data_to_render_list = [ get_Data_to_Render( nSnap, inDir, data_parameters[i], stats=True ) for i in range(nFields)]



#Get Dimensions of the data to render
nz, ny, nx = data_to_render_list[0].shape
nWidth, nHeight, nDepth = nx, ny, nz

#Set the parameters for rendering each field
volumeRender.render_parameters[0] = { 'transp_type':'sigmoid', 'cmap_indx':0, 'transp_center':0, "transp_ramp": 3, 'density':0.03, "brightness":2.0, 'transfer_offset': volumeRender.transfer_offset, 'transfer_scale': volumeRender.transfer_scale }
volumeRender.render_parameters[1] = { 'transp_type':'sigmoid', 'cmap_indx':3, 'transp_center':0, "transp_ramp": 2.5, 'density':0.03, "brightness":2.0, 'transfer_offset': volumeRender.transfer_offset, 'transfer_scale': volumeRender.transfer_scale }
volumeRender.render_parameters[2] = { 'transp_type':'sigmoid', 'cmap_indx':4, 'transp_center':0.0, "transp_ramp": 3, 'density':0.01, "brightness":2.0, 'transfer_offset': volumeRender.transfer_offset, 'transfer_scale': volumeRender.transfer_scale }



#Initialize openGL
volumeRender.width_GL = 512*2
volumeRender.height_GL = 512*2
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
  global exit_program, nSnap, current_frame, rotation_angle, data_for_interpolation, send_data, data_to_render_list, copyToScreen_list
  sendToScreen( )
  if current_frame > 0 and save_images : volumeRender.save_image(dir=outDir, image_name='image')
  if current_frame > 0: rotation_angle += delta_rotation
  volumeRender.Change_Rotation_Angle( rotation_angle )
  if interpolation:
    if exit_program:
      print "Finished Animation" 
      exit()
    current_frame, nSnap = volumeRender.Update_Frame_Number( nSnap, current_frame, frames_per_snapshot )
    if nSnap == n_snapshots-1:
      print( "Exiting")
      exit_program = True
    data_to_render_list, data_for_interpolation = get_Data_List_to_Render_Interpolation( nSnap, inDir, nFields, current_frame, frames_per_snapshot, data_parameters, data_for_interpolation, n_snapshots )
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
