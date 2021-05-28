#!/bin/bash





# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>

cwd=$(pwd)

# Input Images - Single image (224 x 224)
# DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/single_image_224x224_src1"
# DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/single_image_224x224_src2"

# Input Images - Two images (224 x 224)
# DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/two_images_224x224_src1"
# DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/two_images_224x224_src2"

# Input Images - Three images (224 x 224)
DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/three_images_224x224_src1"
DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/three_images_224x224_src2"

# Input Images - Two images (mixed size)
# DEFAULT_SRC_FOLDER_1="$cwd/../TEST_IMAGES/two_images_mixed_src1"
# DEFAULT_SRC_FOLDER_2="$cwd/../TEST_IMAGES/two_images_mixed_src2"

# Input Images - 32 images (224x224)
# DEFAULT_SRC_FOLDER_1="/media/abishek/rppTestImages/224x224_0032_images_src1"
# DEFAULT_SRC_FOLDER_2="/media/abishek/rppTestImages/224x224_0032_images_src1"

# Input Images - 32 images (3840x2160)
# DEFAULT_SRC_FOLDER_1="/media/abishek/rppTestImages/3840x2160_0032_images_src1"
# DEFAULT_SRC_FOLDER_2="/media/abishek/rppTestImages/3840x2160_0032_images_src1"

# Output Images
mkdir "$cwd/../OUTPUT_PERFORMANCE_LOGS_HIP_NEW"
DEFAULT_DST_FOLDER="$cwd/../OUTPUT_PERFORMANCE_LOGS_HIP_NEW"

# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>





# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="$DEFAULT_SRC_FOLDER_1"
SRC_FOLDER_2="$DEFAULT_SRC_FOLDER_2"
DST_FOLDER="$DEFAULT_DST_FOLDER"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>





# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>

rm -rvf "$DST_FOLDER"/*
shopt -s extglob
mkdir build
cd build
rm -rvf ./*
cmake ..
make -j16

printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all PKD3 Inputs..."
echo "##########################################################################################"

mkdir "$DST_FOLDER/PKD3"

printf "\n\nUsage: ./BatchPD_hip_pkd3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:81> <verbosity = 0/1>"

for ((case=0;case<82;case++))
do
printf "\n\n\n\n"
echo "--------------------------------"
printf "Running a New Functionality...\n"
echo "--------------------------------"
for ((bitDepth=0;bitDepth<7;bitDepth++))
do
printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
for ((outputFormatToggle=0;outputFormatToggle<2;outputFormatToggle++))
do

SRC_FOLDER_1_TEMP="$SRC_FOLDER_1"
SRC_FOLDER_2_TEMP="$SRC_FOLDER_2"

mkdir "$DST_FOLDER/PKD3/case_$case"
printf "\n./BatchPD_hip_pkd3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case 0"
# rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/PKD3/case_$case""/case_""$case""_output.csv" ./BatchPD_hip_pkd3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" | tee -a "$DST_FOLDER/BatchPD_hip_pkd3_hip_raw_performance_log.txt"
rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/PKD3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle"".csv" ./BatchPD_hip_pkd3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" | tee -a "$DST_FOLDER/BatchPD_hip_pkd3_hip_raw_performance_log.txt"
echo "------------------------------------------------------------------------------------------"
done
done
done





printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all PLN1 Inputs..."
echo "##########################################################################################"

mkdir "$DST_FOLDER/PLN1"

printf "\n\nUsage: ./BatchPD_hip_pln1 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:81> <verbosity = 0/1>"

for ((case=0;case<82;case++))
do
printf "\n\n\n\n"
echo "--------------------------------"
printf "Running a New Functionality...\n"
echo "--------------------------------"
for ((bitDepth=0;bitDepth<7;bitDepth++))
do
printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
for ((outputFormatToggle=0;outputFormatToggle<1;outputFormatToggle++))
do

SRC_FOLDER_1_TEMP="$SRC_FOLDER_1"
SRC_FOLDER_2_TEMP="$SRC_FOLDER_2"

mkdir "$DST_FOLDER/PLN1/case_$case"
printf "\n./BatchPD_hip_pln1 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case 0"
rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/PLN1/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle"".csv" ./BatchPD_hip_pln1 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" | tee -a "$DST_FOLDER/BatchPD_hip_pln1_hip_raw_performance_log.txt"
echo "------------------------------------------------------------------------------------------"
done
done
done





printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all PLN3 Inputs..."
echo "##########################################################################################"

mkdir "$DST_FOLDER/PLN3"

printf "\n\nUsage: ./BatchPD_hip_pln3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:81> <verbosity = 0/1>"

for ((case=0;case<82;case++))
do
printf "\n\n\n\n"
echo "--------------------------------"
printf "Running a New Functionality...\n"
echo "--------------------------------"
for ((bitDepth=0;bitDepth<7;bitDepth++))
do
printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
for ((outputFormatToggle=0;outputFormatToggle<2;outputFormatToggle++))
do

SRC_FOLDER_1_TEMP="$SRC_FOLDER_1"
SRC_FOLDER_2_TEMP="$SRC_FOLDER_2"

mkdir "$DST_FOLDER/PLN3/case_$case"
printf "\n./BatchPD_hip_pln3 $SRC_FOLDER_1_TEMP $SRC_FOLDER_2_TEMP $bitDepth $outputFormatToggle $case 0"
rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/PLN3/case_$case""/output_case""$case""_bitDepth""$bitDepth""_oft""$outputFormatToggle"".csv" ./BatchPD_hip_pln3 "$SRC_FOLDER_1_TEMP" "$SRC_FOLDER_2_TEMP" "$bitDepth" "$outputFormatToggle" "$case" "0" | tee -a "$DST_FOLDER/BatchPD_hip_pln3_hip_raw_performance_log.txt"
echo "------------------------------------------------------------------------------------------"
done
done
done

# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>