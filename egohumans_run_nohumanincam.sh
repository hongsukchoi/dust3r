#!/bin/bash

## SLURM scripts have a specific format.

### Section1: SBATCH directives to specify job configuration
## %j is the job id, %u is the user id, %A is $SLURM_ARRAY_JOB_ID, %a is $SLURM_ARRAY_TASK_ID

#SBATCH --job-name=egohumans          # Job name
#SBATCH --error=/home/hongsuk/projects/dust3r/jobs/jobs4/%j_%t_%A_%a_log.err           # Error file
#SBATCH --output=/home/hongsuk/projects/dust3r/jobs/jobs4/%j_%t_%A_%a_log.out         # Output file

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --qos=low
#SBATCH --array=0-86
#SBATCH --comment="egohumans with gt focal"
# #SBATCH --signal=B:CONT@5 

### Section 2: Setting environment variables for the job
### Remember that all the module command does is set environment
### variables for the software you need to. Here I am assuming I
### going to run something with python.
### You can also set additional environment variables here and
### SLURM will capture all of them for each task
# !!!!!!!!!!!!!USE ENV WHEN CALLING SBATCH!!!!!!!!!!!


### Signal Handling
trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     # Submit a new job to the queue
     echo "Requeuing " $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
     # SLURM_JOB_ID is a unique representation of the job, equivalent
     # to above
     scontrol requeue $SLURM_JOB_ID
   fi
}

# Install signal handler
trap 'trap_handler USR1' USR1
trap 'trap_handler CONT' CONT
trap 'trap_handler TERM' TERM

# activate conda environment
export PYTHONPATH=/home/hongsuk/projects/dust3r
source /home/hongsuk/anaconda3/etc/profile.d/conda.sh
conda activate dust3r

### Section 3:
### Run your job.
# ## Uses ls /home/achleshwar/amazon_iccv21/meshrcnn/output/ > all_asins_with_3dmodel.txt
# ## python llib/methods/multiview_optimization/launch_main_mvopti.py
IFS=$'\r\n' GLOBIGNORE='*' command eval  'PREFIXES=($(cat ./egohumans_slurm_pred_focal.txt))' # 66


vid_uid=${PREFIXES[$SLURM_ARRAY_TASK_ID]}
# split the string by comma
# save first element in take ($take) and second in capture ($capture)
big_seq=$(echo $vid_uid | cut -d',' -f1)
small_seq=$(echo $vid_uid | cut -d',' -f2)
use_gt_focal=$(echo $vid_uid | cut -d',' -f3)
echo $big_seq 
echo $small_seq
echo $use_gt_focal
# echo $vid_uid

echo $HOSTNAME
GPU_ID=`nvidia-smi --query-gpu=gpu_bus_id --format=csv | tail -n +2`
echo $GPU_ID
## Wait indefinitely if on em3 with bad GPU
if [ "$HOSTNAME" = "em3" ]; then
    if [[ "$GPU_ID" = "00000000:1D:00.0"* ]]; then
        echo "Waiting indefinitely on bad gpu..."
        while true; do sleep 1000; done
    fi
fi

# Check if use_gt_focal is True and construct command accordingly
if [ $use_gt_focal = True ]; then
    echo "Using GroundTruth focal lengths"
    cmd="CUDA_VISIBLE_DEVICES=0 python hongsuk_egohumans_align_dust3r_hmr2hamer_nohumanincam.py --sel-big-seqs $big_seq --sel_small_seq_range $small_seq --use_gt_focal"
else
    cmd="CUDA_VISIBLE_DEVICES=0 python hongsuk_egohumans_align_dust3r_hmr2hamer_nohumanincam.py --sel-big-seqs $big_seq --sel_small_seq_range $small_seq"
fi

# Execute the constructed command
eval $cmd

