import os
import numpy as np
os.environ['OPENCV_IO_ENABLE_JASPER'] = 'True'
from PIL import Image



# There are problems with jp2 decoders so I will use just bash
# and image_magic tools
input_data_path = "Data/R20m"
output_data_path = "Data/R20m_png"
files = os.listdir(input_data_path)
columns_names = []
y = Image.open("Data/R20m/T34UDE_20200815T095039_B02_20m.jp2")
y.save("img1.png")

