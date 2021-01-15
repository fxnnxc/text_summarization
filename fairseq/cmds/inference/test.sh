
DATA_PATH=data/xsum-large
HYPO_NAME=test.hypo2

cd ${DATA_PATH}

cat test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.target.tokenzied
cat ${HYPO_NAME} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${HYPO_NAME}.target
files2rouge ${HYPO_NAME}.target test.target.tokenzied
