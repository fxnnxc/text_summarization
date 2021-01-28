export CLASSPATH=/home/bumjin/fairseq/data/stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar

DATA_PATH=data/xsum-large
NUM_PARALLEL=3
MODEL=checkpoint11_good_stage1_xsum_UF2_LR3e-05_WU300_C256_P1000.pt
TEST_FILE=test.source
TEST_TARGET=test.target
HYPO_NAME=test.hypo_vae_gen1
GPU=7

python inference/bart_vae_inference_parallel.py --model $MODEL \
 --data-path $DATA_PATH\
 --parallel $NUM_PARALLEL\
 --test-file $TEST_FILE\
 --hypo $HYPO_NAME\
 --gpu $GPU;

cd ${DATA_PATH}
cat $TEST_TARGET | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${TEST_TARGET}.tokenized
cat ${HYPO_NAME} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${HYPO_NAME}.target
cat $TEST_FILE | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${TEST_FILE}.tokenized

files2rouge ${HYPO_NAME}.target ${TEST_TARGET}.tokenized
python /home/bumjin/fairseq/inference/ngram2.py ${TEST_FILE}.tokenized ${HYPO_NAME}.target

echo $HYPO_NAME
