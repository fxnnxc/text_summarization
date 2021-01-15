TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500     
LR=3e-05
MAX_TOKENS=2048

DATA_PATH=xsum
BART_PATH=/checkpoints/bart.large.${DATA_PATH}/model.pt

UPDATE_FREQ=4
GPUS=3,4

CHECKPOINT_SUFFIX=${DATA_PATH}BART_LARGE_fp16_UF${UPDATE_FREQ}_GPUS${GPUS}_TEST
ARCH=bart_large
SAVE_INTERVAL=4
MAX_EPOCH=20

CUDA_VISIBLE_DEVICES=${GPUS} python train.py data/${DATA_PATH}-large-bin \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch $ARCH \
    --fp16 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --checkpoint-suffix $CHECKPOINT_SUFFIX \
    --save-interval $SAVE_INTERVAL \
    --max-epoch $MAX_EPOCH ;
