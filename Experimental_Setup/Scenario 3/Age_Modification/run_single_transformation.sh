#!/bin/bash

# It expects three arguments:
# 1. Full path to the moving image (always the subject's 65-year-old scan)
# 2. The age of the fixed image (e.g., 55, 60, 65, 70, 75)
# 3. Full path to the fixed image (subject-specific scan at the target age)

MOVING_IMG_PATH_65YO="$1"
FIXED_IMG_AGE="$2"
FIXED_IMG_PATH_TARGET_AGE="$3"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export PATH="${SCRIPT_DIR}/c3d-nightly-Linux-x86_64/c3d-1.1.0-Linux-x86_64/bin:$PATH"

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 

BASE_IMAGE_DATADIR="/home/ravi.bullock" 

OUTPUT_ROOT_DATADIR="${SLURM_SUBMIT_DIR}/Output_Transformations"

mkdir -p "$OUTPUT_ROOT_DATADIR"


VAR_REG="${HOME}/VariationalRegistration"

SOURCE_AGE=65

if [ ! -f "$MOVING_IMG_PATH_65YO" ]; then
    echo "Error: Moving (65yo) image not found: ${MOVING_IMG_PATH_65YO}. Skipping this task."
    exit 1
fi
if [ ! -f "$FIXED_IMG_PATH_TARGET_AGE" ]; then
    echo "Error: Fixed (target age) image not found: ${FIXED_IMG_PATH_TARGET_AGE}. Skipping this task."
    exit 1
fi
if [ ! -x "$VAR_REG" ]; then
    echo "Error: VariationalRegistration executable not found or not executable: ${VAR_REG}. Skipping this task."
    exit 1
fi
if ! command -v c3d &> /dev/null; then
    echo "Error: c3d command not found after PATH adjustment. Please verify c3d path. Skipping this task."
    exit 1
fi

base_moving_filename=$(basename "$MOVING_IMG_PATH_65YO")
subject_id=$(echo "$base_moving_filename" | grep -oP 'sub-\d+(?=[0-9]{2}_T1w.nii.gz)')
if [ -z "$subject_id" ]; then
    echo "Error: Could not extract subject ID from ${MOVING_IMG_PATH_65YO}. Skipping task."
    exit 1
fi


CURRENT_OUTPUT_DIR="${OUTPUT_ROOT_DATADIR}/${subject_id}_${SOURCE_AGE}_to_${FIXED_IMG_AGE}"
mkdir -p "$CURRENT_OUTPUT_DIR"


outputPrefix="${CURRENT_OUTPUT_DIR}/${base_moving_filename/.nii.gz/}"


converted_moving_img="${outputPrefix}_moving_converted.nii.gz"
echo "[$SLURM_ARRAY_TASK_ID] Converting moving (65yo) image to ushort: $MOVING_IMG_PATH_65YO -> $converted_moving_img"
c3d "$MOVING_IMG_PATH_65YO" -stretch -1 1 0 16000 -type ushort -o "$converted_moving_img"


fixed_template_filename_base=$(basename "$FIXED_IMG_PATH_TARGET_AGE" .nii.gz)
converted_fixed_template_shared="${OUTPUT_ROOT_DATADIR}/templates/${fixed_template_filename_base}_converted.nii.gz"
mkdir -p "${OUTPUT_ROOT_DATADIR}/templates" # Ensure this directory exists

echo "[$SLURM_ARRAY_TASK_ID] Checking/Converting fixed (target age) image to ushort: ${FIXED_IMG_PATH_TARGET_AGE} -> ${converted_fixed_template_shared}"

if [ ! -f "$converted_fixed_template_shared" ]; then
    echo "[$SLURM_ARRAY_TASK_ID] Fixed image template not found in converted cache, converting: ${FIXED_IMG_PATH_TARGET_AGE}"
    c3d "$FIXED_IMG_PATH_TARGET_AGE" -stretch -1 1 0 16000 -type ushort -o "$converted_fixed_template_shared"
else
    echo "[$SLURM_ARRAY_TASK_ID] Converted fixed image template already exists: ${converted_fixed_template_shared}"
fi

echo "[$SLURM_ARRAY_TASK_ID] Reslicing moving image to fixed image space: $converted_moving_img"
c3d "$converted_fixed_template_shared" "$converted_moving_img" -reslice-identity -o "$converted_moving_img"

echo "[$SLURM_ARRAY_TASK_ID] Running VariationalRegistration for ${subject_id} from age ${SOURCE_AGE} to ${FIXED_IMG_AGE}..."
# IMPORTANT: SWAPPED -F (Fixed) and -M (Moving) arguments!
"$VAR_REG" -F "$converted_fixed_template_shared" -M "$converted_moving_img" \
    -O "${outputPrefix}_displ_${SOURCE_AGE}_to_${FIXED_IMG_AGE}.nii.gz" \
    -V "${outputPrefix}_velo_${SOURCE_AGE}_to_${FIXED_IMG_AGE}.nii.gz" \
    -W "${outputPrefix}_warped_${SOURCE_AGE}_to_${FIXED_IMG_AGE}.nii.gz" \
    -p 1 -g 0.0000000000000000001 -i 100000000000 -l 6 -t 1 -x 9 -d 2 -u 1 -h 1 -r 1 -s 2 -f 0 -a 0.2

echo "[$SLURM_ARRAY_TASK_ID] Finished VariationalRegistration for ${subject_id} from age ${SOURCE_AGE} to ${FIXED_IMG_AGE}."

# Clean up temporary converted moving image after processing
#rm -f "$converted_moving_img"
