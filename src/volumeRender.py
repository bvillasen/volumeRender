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
sys.path.extend( [myToolsDirectory, volRenderDirectory] )
from cudaTools import np3DtoCudaArray, np2DtoCudaArray

nWidth = 128
nHeight = 128
nDepth = 128
#nData = nWidth*nHeight*nDepth

windowTitle = "CUDA 3D volume render"


viewXmin, viewXmax = -0.5, 0.5
viewYmin, viewYmax = -0.5, 0.5
viewZmin, viewZmax = -10.5, 10.5



plotData_h = np.random.rand(nWidth*nHeight*nDepth)



def stepFunc():
  print "Default step function"

width_GL = 512*4
height_GL = 512*4

def save_image(dir='', image_name='image'):
  global n_image
  glPixelStorei(GL_PACK_ALIGNMENT, 1)
  width = width_GL
  if nTextures == 2 :width = 2*width_GL
  data = glReadPixels(0, 0, width, height_GL, GL_RGBA, GL_UNSIGNED_BYTE)
  image = Image.frombytes("RGBA", (width, height_GL), data)
  image = ImageOps.flip(image) # in my case image is flipped top-bottom for some reason
  image_file_name = '{0}_{1}.png'.format(image_name, n_image)
  image.save(dir+image_file_name, 'PNG')
  n_image += 1
  print 'Image saved: {0}'.format(image_file_name)



dataMax = plotData_h.max()
plotData_h = (256.*plotData_h/dataMax).astype(np.uint8).reshape(nDepth, nHeight, nWidth)
plotData_dArray = None
plotData_dArray_1 = None
transferFuncArray_d = None

viewRotation =  np.zeros(3).astype(np.float32)
viewTranslation = np.array([0., 0., -3.5])
invViewMatrix_h = np.arange(12).astype(np.float32)
invViewMatrix_h_1 = np.arange(12).astype(np.float32)
scaleX = 1.
separation = 0.

density = 0.05
brightness = 2.0
transferOffset = 0.0
transferScale = 1.0

n_image = 0

#linearFiltering = True
def sigmoid( x, center, ramp ):
  return 1./( 1 + np.exp(-ramp*(x-center)))

def gaussian( x, center, ramp ):
  return np.exp(-(x-center)*(x-center)/ramp/ramp)



transp_type = 'sigmoid'
colorMap = 'jet'
colorMaps = [ 'CMRmap', 'jet', 'nipy_spectral', 'viridis',  'inferno', 'bone', 'hot', 'copper', 'jet']
# colorMaps = plt.colormaps()
cmap_indx_0 = 4
trans_center_0 = np.float32(0)
trans_ramp_0 = np.float32(1)

cmap_indx_1 = 0
trans_center_1 = np.float32(0)
trans_ramp_1 = np.float32(1)



separation = np.float32( separation)
density = np.float32(density)
brightness = np.float32(brightness)
transferOffset = np.float32(transferOffset)
transferScale = np.float32(transferScale)



block2D_GL = (16, 16, 1)
grid2D_GL = (width_GL/block2D_GL[0], height_GL /block2D_GL[1] )

gl_tex = []
nTextures = 1
gl_PBO = []
#nPBOs = 1
cuda_PBO = []


frameCount = 0
fpsCount = 0
fpsLimit = 8
timer = 0.0

#CUDA device variables
c_invViewMatrix = None

#CUDA Kernels
renderKernel = None


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
    # if fpsCount_1 == 2:
    #     viewRotation[1] += np.float32(1)
    #     fpsCount_1 = 0
def render():
  global invViewMatrix_h, invViewMatrix_h_1, c_invViewMatrix
  global gl_PBO, cuda_PBO
  global width_GL, height_GL, density, brightness, transferOffset, transferScale
  global block2D_GL, grid2D_GL
  global tex, transferTex
  global testData_d
  for i in range(nTextures):
    if i == 0:
      set_transfer_function( cmap_indx_0, trans_ramp_0, trans_center_0 )
      # brightness = np.float32(1.0)
      cuda.memcpy_htod( c_invViewMatrix,  invViewMatrix_h)
      tex.set_array(plotData_dArray)
    if i == 1:
      set_transfer_function( cmap_indx_1, trans_ramp_1, trans_center_1 )
      # brightness = np.float32(2)
      cuda.memcpy_htod( c_invViewMatrix,  invViewMatrix_h_1)
      tex.set_array(plotData_dArray_1)
    # map PBO to get CUDA device pointer
    cuda_PBO_map = cuda_PBO[i].map()
    cuda_PBO_ptr, cuda_PBO_size = cuda_PBO_map.device_ptr_and_size()
    cuda.memset_d32( cuda_PBO_ptr, 0, width_GL*height_GL )
    renderKernel( np.intp(cuda_PBO_ptr), np.int32(width_GL), np.int32(height_GL), density, brightness, transferOffset, transferScale, grid=grid2D_GL, block = block2D_GL, texrefs=[tex, transferTex] )
    cuda_PBO_map.unmap()

def get_model_view_matrix( indx=0 ):
  modelView = np.ones(16)
  glMatrixMode(GL_MODELVIEW)
  glPushMatrix()
  glLoadIdentity()
  glRotatef(-viewRotation[0], 1.0, 0.0, 0.0)
  glRotatef(-viewRotation[1], 0.0, 1.0, 0.0)
  if indx == 1:
    glRotatef(-separation, 0.0, 1.0, 0.0)
    # glTranslatef(-separation/2., 0.0, 0.0 )
  if indx == 2:
    glRotatef( separation, 0.0, 1.0, 0.0)
    # glTranslatef( separation/2., 0.0, 0.0 )
  glTranslatef(-viewTranslation[0], -viewTranslation[1], -viewTranslation[2])

  modelView = glGetFloatv(GL_MODELVIEW_MATRIX )
  modelView_copy = modelView.copy()
  modelView = modelView.reshape(16).astype(np.float32)
  # print modelView
  glPopMatrix()
  return modelView


def display():
  global viewRotation, viewTranslation, invViewMatrix_h, invViewMatrix_h_1
  global timer

  timer = time.time()
  stepFunc()

  if nTextures == 1:
    modelView = get_model_view_matrix()
    invViewMatrix_h[0] = -modelView[0]/scaleX
    invViewMatrix_h[1] = -modelView[4]
    invViewMatrix_h[2] = -modelView[8]
    invViewMatrix_h[3] = -modelView[12]
    invViewMatrix_h[4] = -modelView[1]
    invViewMatrix_h[5] = -modelView[5]
    invViewMatrix_h[6] = -modelView[9]
    invViewMatrix_h[7] = -modelView[13]
    invViewMatrix_h[8] = -modelView[2]
    invViewMatrix_h[9] = -modelView[6]
    invViewMatrix_h[10] = -modelView[10]
    invViewMatrix_h[11] = -modelView[14]

  if nTextures == 2:
    modelView = get_model_view_matrix(1)
    invViewMatrix_h[0] = modelView[0]/scaleX
    invViewMatrix_h[1] = modelView[4]
    invViewMatrix_h[2] = modelView[8]
    invViewMatrix_h[3] = modelView[12]
    invViewMatrix_h[4] = modelView[1]
    invViewMatrix_h[5] = modelView[5]
    invViewMatrix_h[6] = modelView[9]
    invViewMatrix_h[7] = modelView[13]
    invViewMatrix_h[8] = modelView[2]
    invViewMatrix_h[9] = modelView[6]
    invViewMatrix_h[10] = modelView[10]
    invViewMatrix_h[11] = modelView[14]
    # invViewMatrix_h = invViewMatrix_h

    modelView = get_model_view_matrix(2)
    invViewMatrix_h_1[0] = modelView[0]/scaleX
    invViewMatrix_h_1[1] = modelView[4]
    invViewMatrix_h_1[2] = modelView[8]
    invViewMatrix_h_1[3] = modelView[12]
    invViewMatrix_h_1[4] = modelView[1]
    invViewMatrix_h_1[5] = modelView[5]
    invViewMatrix_h_1[6] = modelView[9]
    invViewMatrix_h_1[7] = modelView[13]
    invViewMatrix_h_1[8] = modelView[2]
    invViewMatrix_h_1[9] = modelView[6]
    invViewMatrix_h_1[10] = modelView[10]
    invViewMatrix_h_1[11] = modelView[14]

    # invViewMatrix_h_1 = invViewMatrix_h.copy()
  # print modelView
  # invViewMatrix_h = modelView[:12]
  # invViewMatrix_h = modelView.copy()


  render()
   # display results
  glClear(GL_COLOR_BUFFER_BIT)

  for i in range(nTextures):

    # draw image from PBO
    #glDisable(GL_DEPTH_TEST)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    # draw using texture
    # copy from pbo to texture
    glBindBufferARB( GL_PIXEL_UNPACK_BUFFER, gl_PBO[i])
    glBindTexture(GL_TEXTURE_2D, gl_tex[i])
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_GL, height_GL, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, 0)
    # draw textured quad
    #glEnable(GL_TEXTURE_2D)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    if nTextures == 2:
      if i==0: glVertex2f(-1., -0.5)
      if i==1: glVertex2f( 0., -0.5)
    else:  glVertex2f(-0.5, -0.5)
    glTexCoord2f(1, 0)
    if nTextures == 2:
      if i==0: glVertex2f( 0., -0.5)
      if i==1: glVertex2f( 1., -0.5)
    else: glVertex2f(0.5, -0.5)
    glTexCoord2f(1, 1)
    if nTextures == 2:
      if i==0: glVertex2f( 0., 0.5)
      if i==1: glVertex2f( 1., 0.5)
    else: glVertex2f(0.5, 0.5)
    glTexCoord2f(0, 1)
    if nTextures == 2:
      if i==0: glVertex2f(-1., 0.5)
      if i==1: glVertex2f( 0., 0.5)
    else: glVertex2f(-0.5, 0.5)
    glEnd()

    # glDisable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)
    #
    # # modelView = np.ones(16)
    # glMatrixMode(GL_MODELVIEW)
    # glPushMatrix()
    # glLoadIdentity()
    # glTranslatef(-viewTranslation[0], -viewTranslation[1], -viewTranslation[2])
    # # glRotatef(viewRotation[0], 1.0, 0.0, 0.0)
    # glRotatef(-viewRotation[1], 0.0, 1.0, 0.0)
    # # print viewTranslation
    # # modelView = glGetFloatv(GL_MODELVIEW_MATRIX )
    # # modelView = modelView.reshape(16).astype(np.float32)
    # # new_matrix = np.eye(4,4).astype(np.float32)
    # # new_matrix = new_matrix.reshape(16)
    # # new_matrix[0] = invViewMatrix_h[0]/scaleX
    # # new_matrix[4] = invViewMatrix_h[1]
    # # new_matrix[8] = invViewMatrix_h[2]
    # # new_matrix[12] = invViewMatrix_h[3]
    # # new_matrix[1] = invViewMatrix_h[4]
    # # new_matrix[5] = invViewMatrix_h[5]
    # # new_matrix[9] = invViewMatrix_h[6]
    # # new_matrix[13] = invViewMatrix_h[7]
    # # new_matrix[2] = invViewMatrix_h[8]
    # # new_matrix[6] = invViewMatrix_h[9]
    # # new_matrix[10] = invViewMatrix_h[10]
    # # new_matrix[14] =  invViewMatrix_h[11]
    # # print new_matrix
    # # new_matrix = new_matrix.reshape(4,4)
    # # glLoadMatrixf(new_matrix)
    # glBegin(GL_LINES);
    # glVertex3f(-0.5, 0.5, 0 );
    # glVertex3f(0.5, 0.5,  0);
    # glEnd();
    # glPopMatrix()



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
  #glew.glewInit()
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
  #print "Buffer Created"
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
  #print "Texture Created"





def set_transfer_function( cmap_indx, transp_ramp, transp_center ):
  # global transp_ramp, transp_center
  colorMap = colorMaps[cmap_indx]
  norm = cl.Normalize(vmin=0, vmax=1, clip=False)
  cmap = cm.ScalarMappable( norm=norm, cmap=colorMap)
  colorVals = np.linspace(1,0,257)
  colorData = cmap.to_rgba(colorVals).astype(np.float32)
  transp_vals = np.linspace(1,-1,257)
  if transp_type=='sigmoid':transparency = sigmoid( transp_vals, transp_center, transp_ramp )
  if transp_type=='gaussian':transparency = gaussian( transp_vals, transp_center, transp_ramp )
  colorData[:,3] = (colorVals**transp_ramp).astype(np.float32)
  colorData[:,3] = (transparency ).astype(np.float32)
  colorData[0,:] = 1

  # print colorMap, transp_ramp, transp_center


  # transferFunc = np.array([
  #   [  1.0, 0.0, 0.0, 1.0, ],
  #   [  1.0, 0.0, 0.0, 1.0, ],
  #   [  1.0, 0.5, 0.0, 1.0, ],
  #   [  1.0, 1.0, 0.0, 1.0, ],
  #   [  0.0, 1.0, 0.0, 1.0, ],
  #   [  0.0, 1.0, 1.0, 1.0, ],
  #   [  0.0, 0.0, 1.0, 1.0, ],
  #   [  0.4, 0.0, 1.0, 1.0, ],
  #   [  0.0, 0.0, 0.0, 0.0, ]]).astype(np.float32)
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

  if not plotData_dArray: plotData_dArray = np3DtoCudaArray( plotData_h )
  tex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
  tex.set_filter_mode(cuda.filter_mode.LINEAR)
  tex.set_address_mode(0, cuda.address_mode.CLAMP)
  tex.set_address_mode(1, cuda.address_mode.CLAMP)
  tex.set_array(plotData_dArray)

  set_transfer_function( cmap_indx_0, trans_ramp_0, trans_center_0 )
  print "CUDA volumeRender initialized\n"


def keyboard(*args):
  global transferScale, brightness, density, transferOffset, cmap_indx_0
  global separation, trans_center_0, trans_ramp_0
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
    trans_center_0 += np.float32(0.05)
    print trans_center_0
  if args[0] == 'c':
    trans_center_0 -= np.float32(0.05)
    print trans_center_0
  if args[0] == 'f':
    trans_ramp_0 += np.float32(0.05)
    print trans_ramp_0
  if args[0] == 'v':
    trans_ramp_0 -= np.float32(0.05)
    print trans_ramp_0

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
  #initGL()
  #import pycuda.gl.autoinit
  initCUDA()
  initPixelBuffer()
  startGL()
