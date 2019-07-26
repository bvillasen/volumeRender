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
inDir = dataDir + 'cosmo_sims/cholla_pm/256_dm_50Mpc/data/'

#Select CUDA Device
useDevice = 0

nFields = 1

n_snapshots = 259
snapshots = range(0, 259, 1) 

n_frames = 1200
frames_per_snapshot = n_frames / n_snapshots

data_format = 'cholla'
data_type = 'particles'
data_field = 'density'
normalization = 'global'
log_data = True
n_border = 3

nSnap = 0





data_dic = get_data( nSnap, inDir, data_format, data_type, data_field, stats=True )
data_to_render = data_dic['data']
stats = data_dic['stats']

plotData_0 = prepare_data( data_to_render, log=log_data, normalize=normalization, stats=stats, n_border=3)

def Change_Snapshot_Single_Field( nSnap, field_index, copyToScreen_list, inDir, data_format, data_type, data_field, stats=False, log=False, normalization='local', n_border=3  ):
  plotData = get_Data_to_Render( nSnap, inDir, data_format, data_type, data_field, stats=stats, log=log, normalize=normalization, n_border=n_border )
  copyToScreen = copyToScreen_list[field_index]
  copyToScreen.set_src_host(plotData)
  copyToScreen()
  
  

data_to_render_list = [ plotData_0 ]
nz, ny, nx = data_to_render_list[0].shape
nWidth, nHeight, nDepth = nx, ny, nz

volumeRender.render_parameters[0] = { 'transp_type':'sigmoid', 'cmap_indx':0, 'transp_center':0, "transp_ramp": 2.5, 'density':0.03, "brightness":2.0, 'transfer_offset': volumeRender.transfer_offset, 'transfer_scale': volumeRender.transfer_scale }

# Count the frames to change snapshot
nFrame = 0

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
      copyToScreen_list[i]
    send_data = False 
########################################################################

def stepFunction():
  global  nSnap, nFrame
  sendToScreen( )
  nFrame += 1
  if nFrame % frames_per_snapshot == 0:
    nSnap += 1
    print "Change Snapshot: {0}  Frame:{1}".format( nSnap, nFrame)
    Change_Snapshot_Single_Field( nSnap, 0, copyToScreen_list, inDir, data_format, data_type, data_field, stats=True, log=log_data, normalization=normalization, n_border=n_border  )

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
