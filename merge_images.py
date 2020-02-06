import sys, os
from PIL import Image, ImageDraw, ImageFont

currentDirectory = os.getcwd()
srcDirectory = currentDirectory + "/src/"
dataDirectory = currentDirectory + "/data_src/"
sys.path.extend([ srcDirectory, dataDirectory ] )
from tools import create_directory


color_blue = (102, 153, 255)
color_orange = (255, 153, 0)
color_purple = (0, 191, 255)

data_dir = '/home/bruno/Desktop/ssd_0/data/cosmo_sims/1024_hydro_50Mpc/snapshots_pchw18/'
input_dir_0 = data_dir + 'particles_density/render_dm/'
input_dir_1 = data_dir + 'hydro_density/render_hydro/'
output_dir = data_dir + 'render_dm_gas/'
create_directory( output_dir )
# 
n_frame = 0
image_name = 'image_{0}.png'.format(n_frame)
images = [Image.open(x) for x in [ input_dir_0 + image_name, input_dir_0 + image_name ]]
widths, heights = zip(*(i.size for i in images))


img_black = Image.new('RGB', (512, 128), color = (0, 0, 0)) 
 
img_text = Image.new('RGB', (512, 128), color = (0, 0, 0))
text_img = ImageDraw.Draw(img_text)
fnt = ImageFont.truetype('/Library/Fonts/Helvetica.ttf', 50)
text_img.text((256,64), r"z = 100", font=fnt, fill=color_purple)
# img_text.save( output_dir + 'pil_text.png')

img_title_0 = Image.new('RGB', (1024, 128), color = (0, 0, 0)) 
title_0 = ImageDraw.Draw(img_title_0)
fnt = ImageFont.truetype('/Library/Fonts/Helvetica.ttf', 50)
title_0.text((64,64), r"Dark Matter Density", font=fnt, fill=color_blue)


##Merge Images 
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
  im.paste( img_black, (0, 0) )
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.paste( img_text, (0, 0 ))

new_im.paste( img_title_0, (750, 0 ))

new_im.save(output_dir + image_name)