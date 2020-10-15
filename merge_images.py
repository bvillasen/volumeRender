import sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

currentDirectory = os.getcwd()
srcDirectory = currentDirectory + "/src/"
dataDirectory = currentDirectory + "/data_src/"
sys.path.extend([ srcDirectory, dataDirectory ] )
from tools import create_directory


color_blue_dark = (102, 153, 255)
color_orange = (255, 153, 0)
color_blue = (0, 191, 255)

data_dir = '/home/bruno/Desktop/ssd_0/data/cosmo_sims/1024_hydro_50Mpc/snapshots_pchw18/'
input_dir_0 = data_dir + 'render_dm/'
input_dir_1 = data_dir + 'render_hydro/'
# input_dir_0 = data_dir + 'render_hydro/'
# input_dir_1 = data_dir + 'render_temperature/'
output_dir = data_dir + 'render_dm_gas/'
# output_dir = data_dir + 'render_gas_temperature_new/'
create_directory( output_dir )


n_frames = 1999
a_start, a_end = 1./(100+1), 1
a_vals = np.linspace( a_start, a_end, n_frames )
z_vals = 1./a_vals - 1

# n_frames = 1
# n_frame = 500
for n_frame in range(n_frames ):
# for n_frame in [500] :
  image_name = 'image_{0}.png'.format(n_frame)
  images = [Image.open(x) for x in [ input_dir_0 + image_name, input_dir_1 + image_name ]]
  # images = [Image.open(x) for x in [ input_dir_1 + image_name, input_dir_2 + image_name ]]
  widths, heights = zip(*(i.size for i in images))


  img_black = Image.new('RGB', (256, 128), color = (0, 0, 0)) 
   
  img_text = Image.new('RGB', (540, 128), color = (0, 0, 0))
  text_img = ImageDraw.Draw(img_text)
  fnt = ImageFont.truetype('/Library/Fonts/Helvetica.ttf', 80)
  z_val = z_vals[n_frame]
  if z_val > 1: text = 'z = {0:.1f}'.format( z_val )
  else: text = 'z = {0:.2f}'.format( z_val )
  text_img.text((128,64), text, font=fnt, fill=color_blue)
  # img_text.save( output_dir + 'pil_text.png')


  img_title_1 = Image.new('RGB', (1024, 128), color = (0, 0, 0)) 
  title_1 = ImageDraw.Draw(img_title_1)
  fnt = ImageFont.truetype('/Library/Fonts/Helvetica.ttf', 80)
  title_1.text((0,50), r"DM Density", font=fnt, fill=color_blue_dark)

  img_title_2 = Image.new('RGB', (1024, 128), color = (0, 0, 0)) 
  title_2 = ImageDraw.Draw(img_title_2)
  fnt = ImageFont.truetype('/Library/Fonts/Helvetica.ttf', 80)
  title_2.text((0,50), r"Gas Density", font=fnt, fill=color_blue_dark)
  
  # img_title_1 = Image.new('RGB', (1024, 128), color = (0, 0, 0)) 
  # title_1 = ImageDraw.Draw(img_title_1)
  # fnt = ImageFont.truetype('/Library/Fonts/Helvetica.ttf', 80)
  # title_1.text((0,50), r"Gas Density", font=fnt, fill=color_blue_dark)
  # 
  # img_title_2 = Image.new('RGB', (1024, 128), color = (0, 0, 0)) 
  # title_2 = ImageDraw.Draw(img_title_2)
  # fnt = ImageFont.truetype('/Library/Fonts/Helvetica.ttf', 80)
  # title_2.text((0,50), r"Gas Temperature", font=fnt, fill=color_blue_dark)

  ##Merge Images 
  total_width = sum(widths)
  max_height = max(heights)
  y_offset = 90
  new_im = Image.new('RGB', (total_width, max_height + y_offset))
  x_offset = 0
  for im in images:
    im.paste( img_black, (0, 0) )
    new_im.paste(im, (x_offset,y_offset))
    x_offset += im.size[0]

  new_im.paste( img_text, (0, 0 ))

  # new_im.paste( img_title_0, (750, 0 ))
  # new_im.paste( img_title_1, (850+ im.size[0], 0 ))

  new_im.paste( img_title_1, (750, 0 ))
  new_im.paste( img_title_2, (750+ im.size[0], 0 ))


  new_im.save(output_dir + image_name)