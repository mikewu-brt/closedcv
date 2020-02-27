"""
 * Copyright (c) 2018, The LightCo
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are strictly prohibited without prior permission of
 * The LightCo.
 *
 * @author  Bhaskar Mukherjee
 * @version V1.0.0
 * @date    Feb 2020
 * @brief
 *  Gather and rename all captures into a single dir for calibration
 *
 ******************************************************************************/
"""
import os
import shutil
import glob
import importlib
import argparse
import sys

####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Rename all images for calibration")
parser.add_argument('--src_image_dir', default='Oct2_cal')
parser.add_argument('--dst_image_dir', default='images_for_cal')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

try:
  os.makedirs(args.dst_image_dir)
except IOError as e:
  print("Unable to create dir: %s", e)
####################

# Search directory for all individual capture directories and related JSON
list_subfolders = [f.name for f in sorted(os.scandir(args.src_image_dir), key=lambda e: e.name) if f.is_dir()]
for i, dirname in enumerate(list_subfolders):
  print(dirname)
  if dirname.startswith("CaptureRecord"):
    file_pattern = args.src_image_dir + dirname + "/*.bin"
    list_images = glob.glob(file_pattern)
    for filename in (list_images):
      path, filename_only = os.path.split(filename)
      file_prefix, file_suffix = filename_only.rsplit('_', 1)
      new_filename = args.dst_image_dir + "/" + file_prefix + "_f" + str(i) + ".bin"
      print(new_filename)
      try:
        shutil.copy2(filename, new_filename)
      except IOError as e:
        print("Unable to copy file: %s", e)

