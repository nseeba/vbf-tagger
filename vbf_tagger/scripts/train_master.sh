#!/bin/bash
# This script trains the model for the vbf-tagger

PROGNAME=$0

# Parse user options.
usage() {
  cat << EOF >&2
Usage: bash scripts/$PROGNAME [-o] [-s] [-d] [-e] [-m] [-c]
  -o : This is used to specify the output directory.
  -e : Evaluation run [Default: False]
  -m : [OPTIONAL] Use this flag to run the training on 'manivald'. By default it is run on LUMI
EOF
  exit 1
}

EVALUATION=False
HOST=lumi
BASE_DIR="outputs/run"

while getopts 'o:em:' OPTION; do
  case $OPTION in
    o) BASE_DIR=$OPTARG ;;
    e) EVALUATION=True ;;
    m) HOST=manivald ;;
    ?) usage ;;
  esac
done
shift "$((OPTIND - 1))"

echo Output will be saved into: $BASE_DIR
# echo Running on: $HOST
echo Evaluation: $EVALUATION

# if  [ "$HOST" = "lumi" ] ; then
#     TRAINING_SCRIPT=vbf_tagger/scripts/submit_gpu_lumi.sh
# else
#     TRAINING_SCRIPT=vbf_tagger/scripts/submit_gpu_manivald.sh
# fi

# # Apparently Hydra fails to set multilevel config in-place. This is just a workaround to load the correct config.
# sed -i "/clusterization@clusterization.model/ s/: .*/: $CLUSTERIZATION_MODEL/" ml4cc/config/models/two_step/two_step.yaml



# sbatch $TRAINING_SCRIPT python3 vbf_tagger/scripts/train.py training.output_dir=$BASE_DIR environment@host=$HOST training.type=classification training.model_evaluation=$EVALUATION




sbatch vbf_tagger/scripts/submit_gpu_manivald.sh $BASE_DIR $EVALUATION

# python3 vbf_tagger/scripts/train.py \
#     training.output_dir=$BASE_DIR \
#     environment@host=local \
#     training.type=classification \
#     training.model_evaluation=$EVALUATION

# sbatch vbf_tagger/scripts/submit_gpu_manivald.sh \
#     python3 vbf_tagger/scripts/train.py \
#         training.output_dir=$BASE_DIR \
#         environment@host=$HOST \
#         training.type=classification \
#         training.model_evaluation=$EVALUATION