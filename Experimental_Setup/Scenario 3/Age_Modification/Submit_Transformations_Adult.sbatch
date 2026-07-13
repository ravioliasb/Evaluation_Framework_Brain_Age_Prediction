#!/bin/bash
#SBATCH --job-name=UKBB_Velocity
#SBATCH --time=5-12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs_velocity/array_task_%A_%a.out
#SBATCH --error=logs_velocity/array_task_%A_%a.err
#SBATCH --array=0-999  


TASK_LIST_FILE="task_list_adult.txt"


mkdir -p logs


if [ ! -s "$TASK_LIST_FILE" ]; then
    echo "Error: Task list file is empty or not found: $TASK_LIST_FILE"
    exit 1
fi


if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: This script must be submitted as a job array (e.g., sbatch --array=0-N ...)"
    exit 1
fi


TASK_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_LIST_FILE")


MOVING_IMG_PATH_65YO=$(echo "$TASK_LINE" | cut -d',' -f1)
FIXED_IMG_AGE=$(echo "$TASK_LINE" | cut -d',' -f2)
FIXED_IMG_PATH_TARGET_AGE=$(echo "$TASK_LINE" | cut -d',' -f3)

echo "Starting array task $SLURM_ARRAY_TASK_ID:"
echo "  Moving Image (65yo): ${MOVING_IMG_PATH_65YO}"
echo "  Fixed Image Age: ${FIXED_IMG_AGE}"
echo "  Fixed Image Path: ${FIXED_IMG_PATH_TARGET_AGE}"

bash spie_extension_run_single_transformation_adult.sh "$MOVING_IMG_PATH_65YO" "$FIXED_IMG_AGE" "$FIXED_IMG_PATH_TARGET_AGE"
