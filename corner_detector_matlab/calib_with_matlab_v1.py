import subprocess
import argparse
import os



parser = argparse.ArgumentParser(description="calib_with_matlab_v1 ")
parser.add_argument('--in_dir', default='/Users/yhussain/project/cal_images/2020-03-05_calibration_subset')
parser.add_argument('--out_dir', default='/Users/yhussain/project/cal_images/2020-03-05_calibration_subset')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

print("***************************************")
print("Command line options:")
print("  {}".format(args))
print("***************************************")
print("")

matlab_path="/Applications/MATLAB_R2019a.app/bin"

input_dir = args.in_dir
output_dir = args.out_dir
dir_name_py_scripts="."
path_to_inirender="/Users/yhussain/project/compimaging_base/compimaging/bazel-out/darwin-opt/bin/"
exr_dir_basename="matlab_exr_exper_tmp1"

# extract the date of capture from the provided input directory: assumption is that the folders have naming convention of CaptureRecord_#date#_#timeofcapture#
# we will use #date# as marker in output light_header file.
subfolders = [ f.path for f in os.scandir(input_dir) if f.is_dir() ]
signature = []
for subfolder in subfolders:
    print(subfolder)
    if subfolder.find("CaptureRecord_") != -1:
        signature = os.path.basename(subfolder.replace("CaptureRecord_",""))
        break

output_light_header = "light_header_{}.json".format(signature[0:10])


not_factory_calib = True

#step 1: invoke matlab script to detect chessboard

print(["{}/matlab".format(matlab_path), "-batch", "matlab_detect_checkerboard_script({})".format(input_dir)])
completed = subprocess.run(["{}/matlab".format(matlab_path), "-batch", "matlab_detect_checkerboard_script('{}')".format(input_dir)], capture_output=True)
print("*********************************************")
print("Results of Step 1 : Matlab Corner Detector")
print(completed)
if completed.returncode != 0:
    print("Matlab Corner Detector did not Finish Successfully")
    print("*********************************************")
    exit(1)
print("*********************************************")

#step 2 - convert matlab detected points stored as mat file into exr files.
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
completed = subprocess.run(["python3", "mat_to_exr.py", "--input_mat_file", "{}/input_for_extrinsincs_function.mat".format(input_dir),"--output_exr_dir", "{}/{}".format(output_dir, exr_dir_basename), "--boardsize_x",  "19", "--boardsize_y",  "44", "--num_cam",  "4"], capture_output=True)
print("*********************************************")
print("Results of Step 2 : Generation of EXR files")
print(completed)
if completed.returncode != 0:
    print("Step 2 did not Finish Successfully")
    print("*********************************************")
    exit(1)
print("*********************************************")

#
##step 3 - invoke ini renderer for BA optimization : factory calib or bypass mode based on not_factory_calib flag

zero_tang="false"
print("*********************************************")
print("Results of Step 3 : BA Step ")
if not_factory_calib:
    opencv_mode="bypass_opencv_zero_tang_false"
    input_light_header = "corners_exr/{}/light_header.json".format(opencv_mode)
    input_chart_rotations = "corners_exr/{}/chart_rotations.fst".format(opencv_mode)
    input_chart_translations = "corners_exr/{}/chart_translations.fst".format(opencv_mode)
    ini_file = "bypass_opencv_board_deformation_matlab.ini"
    completed = subprocess.run(["{}/depthperc/inirenderer/inirenderer".format(path_to_inirender), "-c", "{}/{}".format(input_dir,ini_file),
                            "-i",  "{}/{}".format(input_dir, input_light_header),
                            "-o", "{}".format(output_light_header), "-o",  "chart_rotations.fst", "-o",
                            "chart_translation.fst", "-o",  "rerror0.fst", "-o", "rerror1.fst", "-o", "rerror2.fst", "-o",
                            "rerror3.fst", "-o", "chart_warp.fst", "-i", "{}/{}".format(input_dir, input_chart_rotations),
                            "-i", "{}/{}".format(input_dir, input_chart_translations), "-i", "{},literal_bool".format(zero_tang),
                            "-i", "{}/{},literal_string".format(output_dir,exr_dir_basename),
                            "-d", "{}/{}/bypass_opencv_zero_tang_{}".format(output_dir, exr_dir_basename,zero_tang) ],
                           capture_output=True)
else:
    ini_file = "factory_calibrate.ini"
    input_light_header="light_header_factory_init.json"
    opencv_only="false"
    completed = subprocess.run(["{}/depthperc/inirenderer/inirenderer".format(path_to_inirender), "-c", "{}/{}".format(input_dir,ini_file),
                                "-i",  "{}/{}".format(input_dir, input_light_header),
                                "-o", "{}".format(output_light_header), "-o",  "chart_rotations.fst", "-o",
                                "chart_translation.fst", "-o",  "rerror0.fst", "-o", "rerror1.fst", "-o", "rerror2.fst", "-o",
                                "rerror3.fst", "-o", "chart_warp.fst",
                                "-i", "{},literal_bool".format(opencv_only), "-i", "{},literal_bool".format(zero_tang),
                                "-i", "{}/{},literal_string".format(output_dir,exr_dir_basename),
                                "-d", "{}/{}/opencv_only_{}_zero_tang_{}".format(output_dir, exr_dir_basename,opencv_only, zero_tang) ],
                               capture_output=True)


    if completed.returncode != 0:
        print("BA step did not Finish Successfully")
        print("*********************************************")
print(completed)
print("*********************************************")
