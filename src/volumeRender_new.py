#!/usr/bin/env python
# -*- coding: utf-8 -*-
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
import numpy as np
import sys, time, os
import pycuda.driver as cuda
import pycuda.gl as cuda_gl
from pycuda.compiler import SourceModule
from pycuda import cumath
import pycuda.gpuarray as gpuarray
import matplotlib.colors as cl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
#import pyglew as glew

from PIL import Image
from PIL import ImageOps

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
devDir = '/home/bruno/Desktop/Dropbox/Developer/pyCUDA/'
myToolsDirectory = currentDirectory + '/src/'
volRenderDirectory = currentDirectory + '/src/'
dataSrcDir = currentDirectory + '/data_src/'
sys.path.extend( [myToolsDirectory, volRenderDirectory, dataSrcDir ] )
from cudaTools import np3DtoCudaArray, np2DtoCudaArray
from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, np3DtoCudaArray
from data_functions import *

nWidth = 128
nHeight = 128
nDepth = 128
#nData = nWidth*nHeight*nDepth

windowTitle = "CUDA 3D volume render"


viewXmin, viewXmax = -0.5, 0.5
viewYmin, viewYmax = -0.5, 0.5
viewZmin, viewZmax = -10.5, 10.5

plotData_list = []


width_GL = 512*2
height_GL = 512*2


density = 0.05
brightness = 2.0
transfer_offset = 0.0
transfer_scale = 1.0



colorMaps = [ 'inferno',  'viridis', 'jet', 'nipy_spectral', 'CMRmap', 'bone', 'hot', 'copper']
# colorMaps = plt.colormaps()


render_parameters = {}


viewRotation =  np.zeros(3).astype(np.float32)
viewTranslation = np.array([0., 0., -3.5])

invViewMatrix_h = np.arange(12).astype(np.float32)
invViewMatrix_h_1 = np.arange(12).astype(np.float32)
transferFuncArray_d = None

#Image Parameters
scaleX = 1.
separation = 0.

#Ouput Imahe number
n_image = 0

# Cuda Parameters
block2D_GL = (32, 32, 1)
grid2D_GL = (width_GL/block2D_GL[0], height_GL /block2D_GL[1] )

#OpenGl components
gl_tex = []
gl_PBO = []
cuda_PBO = []
frameCount = 0
fpsCount = 0
fpsLimit = 8
timer = 0.0

#CUDA device variables
c_invViewMatrix = None

#CUDA Kernels
renderKernel = None


def Initialize_GPU_Data( nFields, data_to_render_list ):
  print "\nInitializing Data"
  initialMemory = getFreeMemory( show=True )
  copyToScreen_list = []
  for i in range(nFields):
    plotData_dArray, copyToScreenArray = np3DtoCudaArray( data_to_render_list[i] )
    plotData_list.append( plotData_dArray)
    copyToScreen_list.append( copyToScreenArray )
  finalMemory = getFreeMemory( show=False )
  print " Total Global Memory Used: {0} Mbytes\n".format(float(initialMemory-finalMemory)/1e6)
  return copyToScreen_list




def save_image(dir='', image_name='image'):
  global n_image
  glPixelStorei(GL_PACK_ALIGNMENT, 1)
  width = nTextures * width_GL
  data = glReadPixels(0, 0, width, height_GL, GL_RGBA, GL_UNSIGNED_BYTE)
  image = Image.frombytes("RGBA", (width, height_GL), data)
  image = ImageOps.flip(image) # in my case image is flipped top-bottom for some reason
  image_file_name = '{0}_{1}.png'.format(image_name, n_image)
  image.save(dir+image_file_name, 'PNG')
  n_image += 1
  print 'Image saved: {0}'.format(image_file_name)




#CUDA Textures
tex = None
transferTex = None
fpsCount_1 = 0
def computeFPS():
  global frameCount, fpsCount, fpsLimit, timer, fpsCount_1
  frameCount += 1
  fpsCount += 1
  fpsCount_1 += 1

  if fpsCount == fpsLimit:
    ifps = 1.0 /timer
    glutSetWindowTitle(windowTitle + "      fps={0:0.2f}".format( float(ifps) ))
    fpsCount = 0

def render(  invViewMatrix_list ):
  global gl_PBO, cuda_PBO
  global width_GL, height_GL, density, brightness, transferOffset, transferScale
  global block2D_GL, grid2D_GL
  global tex, transferTex
  global testData_d
  for i in range(nTextures):
    parameters = render_parameters[i]
    transp_type = parameters['transp_type']
    cmap_indx = parameters['cmap_indx']
    transp_center = parameters['transp_center']
    transp_ramp = parameters['transp_ramp']
    density = parameters['density']
    brightness = parameters['brightness']
    transferOffset = parameters['transfer_offset']
    transferScale = parameters['transfer_scale']
    plot_data = plotData_list[i]
    set_transfer_function( transp_type, cmap_indx, transp_ramp, transp_center )
    cuda.memcpy_htod( c_invViewMatrix,  invViewMatrix_list[i])
    tex.set_array(plot_data)

    # map PBO to get CUDA device pointer
    cuda_PBO_map = cuda_PBO[i].map()
    cuda_PBO_ptr, cuda_PBO_size = cuda_PBO_map.device_ptr_and_size()
    cuda.memset_d32( cuda_PBO_ptr, 0, width_GL*height_GL )
    renderKernel( np.intp(cuda_PBO_ptr), np.int32(width_GL), np.int32(height_GL), np.float32(density), np.float32(brightness), np.float32(transferOffset), np.float32(transferScale), grid=grid2D_GL, block = block2D_GL, texrefs=[tex, transferTex] )
    cuda_PBO_map.unmap()

def get_model_view_matrix( indx=0 ):
  modelView = np.ones(16)
  glMatrixMode(GL_MODELVIEW)
  glPushMatrix()
  glLoadIdentity()
  glRotatef(-viewRotation[0], 1.0, 0.0, 0.0)
  glRotatef(-viewRotation[1], 0.0, 1.0, 0.0)
  #Steroscopic View
  # if indx == 1:
  #   glRotatef(-separation, 0.0, 1.0, 0.0)
  #   # glTranslatef(-separation/2., 0.0, 0.0 )
  # if indx == 2:
  #   glRotatef( separation, 0.0, 1.0, 0.0)
  #   # glTranslatef( separation/2., 0.0, 0.0 )
  glTranslatef(-viewTranslation[0], -viewTranslation[1], -viewTranslation[2])

  modelView = glGetFloatv(GL_MODELVIEW_MATRIX )
  modelView_copy = modelView.copy()
  modelView = modelView.reshape(16).astype(np.float32)
  glPopMatrix()
  return modelView

def get_invViewMatrix( indx=0 ):
  invViewMatrix = np.arange(12).astype(np.float32)
  modelView = get_model_view_matrix( indx )
  if nTextures == 1: modelView *= -1
  invViewMatrix[0] = modelView[0]/scaleX
  invViewMatrix[1] = modelView[4]
  invViewMatrix[2] = modelView[8]
  invViewMatrix[3] = modelView[12]
  invViewMatrix[4] = modelView[1]
  invViewMatrix[5] = modelView[5]
  invViewMatrix[6] = modelView[9]
  invViewMatrix[7] = modelView[13]
  invViewMatrix[8] = modelView[2]
  invViewMatrix[9] = modelView[6]
  invViewMatrix[10] = modelView[10]
  invViewMatrix[11] = modelView[14]
  return invViewMatrix
  
def display():
  global viewRotation, viewTranslation, invViewMatrix_h, invViewMatrix_h_1
  global timer

  timer = time.time()
  stepFunc()
  
  invViewMatrix_list = [ get_invViewMatrix() for i in range(nTextures)]
  #Render the Images
  render( invViewMatrix_list )
  # Display results
  glClear(GL_COLOR_BUFFER_BIT)

  for i in range(nTextures):

    # draw image from PBO
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    # draw using texture
    # copy from pbo to texture
    glBindBufferARB( GL_PIXEL_UNPACK_BUFFER, gl_PBO[i])
    glBindTexture(GL_TEXTURE_2D, gl_tex[i])
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_GL, height_GL, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, 0)

    # draw textured quad
    glBegin(GL_QUADS)
    dx = 0.5 * 2
    Lx = dx * nTextures
    #First Vertex
    vertex_y = -0.5
    vertex_x = -Lx/2 + i*dx  
    glTexCoord2f(0, 0)
    glVertex2f(vertex_x, vertex_y)
    #Seconfd Vertex
    vertex_y = -0.5
    vertex_x = -Lx/2 + (i+1)*dx
    glTexCoord2f(1, 0)
    glVertex2f(vertex_x, vertex_y)
    #Third Vertex
    vertex_y = 0.5
    vertex_x = -Lx/2 + (i+1)*dx
    glTexCoord2f(1, 1)
    glVertex2f(vertex_x, vertex_y)
    #Forth Vertex
    vertex_y = 0.5
    vertex_x = -Lx/2 + i*dx 
    glTexCoord2f(0, 1)
    glVertex2f(vertex_x, vertex_y)
    #Finish Image
    glEnd()
    glBindTexture(GL_TEXTURE_2D, 0)
  glutSwapBuffers();
  timer = time.time() - timer
  computeFPS()


def iDivUp( a, b ):
  if a%b != 0:
    return a/b + 1
  else:
    return a/b



GL_initialized = False
def initGL():
  global GL_initialized
  if GL_initialized: return
  glutInit()
  glutInitDisplayMode(GLUT_RGB |GLUT_DOUBLE )
  glutInitWindowSize(width_GL*nTextures, height_GL)
  #glutInitWindowPosition(50, 50)
  glutCreateWindow( windowTitle )
  GL_initialized = True
  print "OpenGL initialized"

def initPixelBuffer():
  global cuda_PBO
  for i in range(nTextures):
    PBO = glGenBuffers(1)
    gl_PBO.append(PBO)
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, gl_PBO[i])
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER, width_GL*height_GL*4, None, GL_STREAM_DRAW_ARB)
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, 0)
    cuda_PBO.append( cuda_gl.RegisteredBuffer(long(gl_PBO[i])) )
  
  #Create texture which we use to display the result and bind to gl_tex
  glEnable(GL_TEXTURE_2D)

  for i in range(nTextures):
    tex = glGenTextures(1)
    gl_tex.append( tex )
    glBindTexture(GL_TEXTURE_2D, gl_tex[i])
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_GL, height_GL, 0,
		  GL_RGBA, GL_UNSIGNED_BYTE, None);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glBindTexture(GL_TEXTURE_2D, 0)
  

#Transfer Functions
#linearFiltering = True
def sigmoid( x, center, ramp ):
  return 1./( 1 + np.exp(-ramp*(x-center)))

def gaussian( x, center, ramp ):
  return np.exp(-(x-center)*(x-center)/ramp/ramp)



def set_transfer_function( transp_type, cmap_indx, transp_ramp, transp_center ):
  transp_center , transp_ramp = np.float32(transp_center), np.float32(transp_ramp)
  colorMap = colorMaps[cmap_indx]
  norm = cl.Normalize(vmin=0, vmax=1, clip=False)
  cmap = cm.ScalarMappable( norm=norm, cmap=colorMap)
  nSamples = 256
  colorVals = np.linspace(0,1,nSamples)
  colorData = cmap.to_rgba(colorVals).astype(np.float32)
  transp_vals = np.linspace(-1,1,nSamples)
  if transp_type=='sigmoid':transparency = sigmoid( transp_vals, transp_center, transp_ramp )**2
  if transp_type=='gaussian':transparency = gaussian( transp_vals, transp_center, transp_ramp )
  # colorData[:,3] = (colorVals)**2
  colorData[:,3] = (transparency )
  colorData[-1,:] = 1
  transferFunc = colorData.copy()
  transferFuncArray_d, desc = np2DtoCudaArray( transferFunc )
  transferTex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
  transferTex.set_filter_mode(cuda.filter_mode.LINEAR)
  transferTex.set_address_mode(0, cuda.address_mode.CLAMP)
  transferTex.set_address_mode(1, cuda.address_mode.CLAMP)
  transferTex.set_array(transferFuncArray_d)




def initCUDA():
  global plotData_dArray
  global tex, transferTex
  global transferFuncArray_d
  global c_invViewMatrix
  global renderKernel
  #print "Compiling CUDA code for volumeRender"
  cudaCodeFile = open(volRenderDirectory + "/CUDAvolumeRender.cu","r")
  cudaCodeString = cudaCodeFile.read()
  cudaCodeStringComplete = cudaCodeString
  cudaCode = SourceModule(cudaCodeStringComplete, no_extern_c=True, include_dirs=[volRenderDirectory] )
  tex = cudaCode.get_texref("tex")
  transferTex = cudaCode.get_texref("transferTex")
  c_invViewMatrix = cudaCode.get_global('c_invViewMatrix')[0]
  renderKernel = cudaCode.get_function("d_render")

  # if not plotData_dArray: plotData_dArray = np3DtoCudaArray( plotData_h )
  tex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
  tex.set_filter_mode(cuda.filter_mode.LINEAR)
  tex.set_address_mode(2, cuda.address_mode.CLAMP)
  # tex.set_address_mode(1, cuda.address_mode.CLAMP)
  # tex.set_array(plotData_dArray)
  print "CUDA volumeRender initialized\n"


def keyboard(*args):
  global transferScale, brightness, density, transferOffset, cmap_indx_0
  global separation
  ESCAPE = '\033'
  # If escape is pressed, kill everything.
  if args[0] == ESCAPE:
    print "Ending Simulation"
    #cuda.gl.Context.pop()
    sys.exit()
  if args[0] == '1':
    transferScale += np.float32(0.01)
    print "Image Transfer Scale: ",transferScale
  if args[0] == '2':
    transferScale -= np.float32(0.01)
    print "Image Transfer Scale: ",transferScale
  if args[0] == '4':
    brightness -= np.float32(0.01)
    print "Image Brightness : ",brightness
  if args[0] == '5':
    brightness += np.float32(0.01)
    print "Image Brightness : ",brightness
  if args[0] == '7':
    density -= np.float32(0.01)
    print "Image Density : ",density
  if args[0] == '8':
    density += np.float32(0.01)
    print "Image Density : ",density
  if args[0] == '3':
    transferOffset += np.float32(0.01)
    print "Image Offset : ", transferOffset
  if args[0] == '6':
    transferOffset -= np.float32(0.01)
    print "Image Offset : ", transferOffset
  if args[0] == '0':
    cmap_indx_0+=1
    if cmap_indx_0==len(colorMaps): cmap_indx_0=0
  if args[0] == 's': save_image()
  if args[0] == 'a':
    separation += np.float32(0.05)
    print separation
  if args[0] == 'z':
    separation -= np.float32(0.05)
    print separation
  if args[0] == 'd':
    transp_center_0 += np.float32(0.05)
    print transp_center_0
  if args[0] == 'c':
    transp_center_0 -= np.float32(0.05)
    print transp_center_0
  if args[0] == 'f':
    transp_ramp_0 += np.float32(0.05)
    print transp_ramp_0
  if args[0] == 'v':
    transp_ramp_0 -= np.float32(0.05)
    print transp_ramp_0

def specialKeys( key, x, y ):
  if key==GLUT_KEY_UP:
    print "UP-arrow pressed"
  if key==GLUT_KEY_DOWN:
    print "DOWN-arrow pressed"


ox = 0
oy = 0
buttonState = 0
def mouse(button, state, x , y):
  global ox, oy, buttonState

  if state == GLUT_DOWN:
    buttonState |= 1<<button
    if button == 3:  #wheel up
      viewTranslation[2] += 0.5
    if button == 4:  #wheel down
      viewTranslation[2] -= 0.5
  elif state == GLUT_UP:
    buttonState = 0
  ox = x
  oy = y
  glutPostRedisplay()

def motion(x, y):
  global viewRotation, viewTranslation
  global ox, oy, buttonState
  dx = x - ox
  dy = y - oy
  if buttonState == 4:
    viewTranslation[0] += dx/100.
    viewTranslation[1] -= dy/100.
  elif buttonState == 2:
    viewTranslation[0] += dx/100.
    viewTranslation[1] -= dy/100.
  elif buttonState == 1:
    viewRotation[0] += dy/5.
    viewRotation[1] += dx/5.
  ox = x
  oy = y
  glutPostRedisplay()

def reshape(w, h):
  global width_GL, height_GL
  global grid2D_GL, block2D_GL
  initPixelBuffer()
  #width_GL, height_GL = w, h
  grid2D_GL = ( iDivUp(width_GL, block2D_GL[0]), iDivUp(height_GL, block2D_GL[1]) )
  glViewport(0, 0, w, h)

  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  #glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
  if w <= h: glOrtho( viewXmin, viewXmax,
				viewYmin*h/w, viewYmax*h/w,
				viewZmin, viewZmax)
  else: glOrtho(viewXmin*w/h, viewXmax*w/h,
		viewYmin, viewYmax,
		viewZmin, viewZmax)
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()


def startGL():
  glutDisplayFunc(display)
  glutKeyboardFunc(keyboard)
  glutSpecialFunc(specialKeys)
  glutMouseFunc(mouse)
  glutMotionFunc(motion)
  glutReshapeFunc(reshape)
  glutIdleFunc(glutPostRedisplay)
  glutMainLoop()

#OpenGL main
def animate():
  global windowTitle
  print "Starting Volume Render"
  initCUDA()
  initPixelBuffer()
  startGL()


def get_CUDA_threads( block_size_x, block_size_y, block_size_z):
  gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
  gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
  gridz = nDepth // block_size_z + 1 * ( nDepth % block_size_z != 0 )
  block3D = (block_size_x, block_size_y, block_size_z)
  grid3D = (gridx, gridy, gridz)
  return grid3D, block3D

#Read and compile CUDA code
print "\nCompiling CUDA code"
########################################################################
from pycuda.elementwise import ElementwiseKernel
floatToUchar = ElementwiseKernel(arguments="float *input, unsigned char *output",
				operation = "output[i] = (unsigned char) ( -255*(input[i]-1));",
				name = "floatToUchar_kernel")
########################################################################


def Change_Rotation_Angle( new_angle ):
  global viewRotation
  viewRotation[1] = new_angle 
  

def Update_Frame_Number( nSnap, current_frame, frames_per_snapshot ):
  current_frame += 1
  if current_frame % frames_per_snapshot == 0: nSnap += 1
  return current_frame, nSnap



def Change_Snapshot_Single_Field( nSnap, field_index, copyToScreen_list, inDir, data_parameters, stats=False  ):
  plotData = get_Data_to_Render( nSnap, inDir, data_parameters, stats=stats )
  copyToScreen = copyToScreen_list[field_index]
  copyToScreen.set_src_host(plotData)
  copyToScreen()
  
def Change_Data_to_Render( nFields, data_to_render_list, copyToScreen_list ):
  for i in range(nFields):
    plotData = data_to_render_list[i]
    copyToScreen = copyToScreen_list[i]
    copyToScreen.set_src_host(plotData)
    copyToScreen()
  
