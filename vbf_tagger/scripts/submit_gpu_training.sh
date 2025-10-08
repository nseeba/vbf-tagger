#!/bin/bash

PROGNAME=$0

# Parse user options.
usage() {
  cat << EOF >&2
Usage: bash enreg/scripts/$PROGNAME [-o] [-c] [-p] [-m]
  -o : This is used to specify the output directory.
  -c : Train classification
  -m : [OPTIONAL] Use this flag to run the training on 'manivald'. By default it is run on LUMI
EOF
  exit 1
}

OUTPUT_EXISTS=false
RUN_ON_LUMI=true
TRAIN_CLASSIFICATION=false

while getopts 'o:cpsm' OPTION; do
  case $OPTION in
    o)
        BASE_DIR=$OPTARG
        OUTPUT_EXISTS=true
        ;;
    c) TRAIN_CLASSIFICATION=true ;;
    m) RUN_ON_LUMI=false ;;
    ?) usage ;;
  esac
done
shift "$((OPTIND - 1))"

echo Output will be saved into: $BASE_DIR
echo Training classification: $TRAIN_CLASSIFICATION

if  [ "$RUN_ON_LUMI" = true ] ; then
    TRAINING_SCRIPT=vbf_tagger/scripts/submit-gpu-lumi.sh
else
    TRAINING_SCRIPT=vbf_tagger/scripts/submit-gpu-manivald.sh
fi

if [ "$TRAIN_CLASSIFICATION" = true ] ; then
    sbatch $TRAINING_SCRIPT python3 vbf_tagger/scripts/train.py training.output_dir=$BASE_DIR
fi