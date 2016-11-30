#!/bin/bash
# Usage:
# ./experiments/scripts/rfcn_end2end_ohem.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/rfcn_end2end_ohem_ucd.sh 1 ResNet-50 ucar_driveless \
#   output/rfcn_ohem_ucd/ucd_nobg_trainval/resnet50_rfcn_ohem_iter_10000.solverstate \
#   --set EXP_DIR rfcn_ohem_ucd RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3
SNAPSHOT=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  ucar_driveless)
    TRAIN_IMDB="ucd_nobg_trainval"
    TEST_IMDB="ucd_nobg_test"
    PT_DIR="ucar_driveless"
    ITERS=160000
    ;;
  pascal_voc)
    TRAIN_IMDB="voc_0712_trainval"
    TEST_IMDB="voc_0712_test"
    PT_DIR="pascal_voc"
    ITERS=110000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_val"
    PT_DIR="coco"
    ITERS=960000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/rfcn_end2end_ohem_ucd_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/rfcn_end2end/solver_ohem.prototxt \
  --snapshot ${SNAPSHOT} \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/rfcn_end2end_ohem_ucd.yml \
  ${EXTRA_ARGS}

#  --weights data/imagenet_models/${NET}-model.caffemodel \

set +x
#NET_FINAL=`tail -n 100 ${LOG} | grep -B 1 "done solving" | grep "Wrote snapshot" | awk '{print $4}'`
NET_FINAL=`grep -B 2 "done solving" ${LOG} | grep "Snapshotting to binary proto file" | awk '{print $10}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/rfcn_end2end/test_agnostic.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/rfcn_end2end_ohem_ucd.yml \
  ${EXTRA_ARGS}
