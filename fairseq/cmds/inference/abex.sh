DATA_PATH=data/xsum-large
MODEL=checkpoint_bestxsum_abex1_UF4_GPUS3_a1_C125.pt
NUM_PARALLEL=3
TEST_FILE=test.source
HYPO_NAME=test.hypo_checkpoint_bestxsum_abex1_UF4_GPUS3_a1_C125.pt
GPU=7

python inference/bart_vae_inference_parallel.py --model $MODEL \
 --data-path $DATA_PATH\
 --parallel $NUM_PARALLEL\
 --test-file $TEST_FILE\
 --hypo $HYPO_NAME\
 --gpu $GPU;

cd ${DATA_PATH}

cat test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.target.tokenzied
cat ${HYPO_NAME} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${HYPO_NAME}.target
files2rouge ${HYPO_NAME}.target test.target.tokenzied

