from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

n = 20 #　Number of expansions per image

# Input Save path
input_path = "dir_name"
files = glob.glob(input_path + '/*.jpg')

# Output Save Path
output_path = "dir_name"
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)

for i, f in enumerate(files):

  img = load_img(f)
  x = img_to_array(img)
  x = np.expand_dims(x, axis=0)

  # Create ImageDataGenerator
  datagen = ImageDataGenerator(
    zca_epsilon=1e-06,   # Epsilon for whitening
    rotation_range=10.0, # Randomly rotated range
    width_shift_range=0.0, # Random width shift range
    height_shift_range=0.0, # Randomly shifted height range
    brightness_range=None, # Range of random brightness shift
    zoom_range=0.0,        # Random zoom range
    horizontal_flip=True, # Randomly flips horizontally
    vertical_flip=True, # Randomly flips vertically
  )

  # Data Augmentation of n images per sheet
  dg = datagen.flow(x, batch_size=1, save_to_dir=output_path, save_prefix='img', save_format='jpg')
  for i in range(n):
    batch = dg.next()