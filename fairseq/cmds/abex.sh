GPUS=1,2,3,4,6
LATENT_ALPHA=0.9
LATENT_DIM=256
PERIOD=1000
ADDITIONAL=1_good

DATA_PATH=xsum
UPDATE_FREQ=2
VERSION=stage1
ARCH=CVAE_large
TASK=simple_vae_translation
CRITERION=kld_loss

BART_PATH=checkpoints/bart.large.${DATA_PATH}/model.pt
#BART_PATH=checkpoints/bart.large/model.pt
USERDIR=custom/${VERSION}
TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=300
LR=3e-05
MAX_TOKENS=2000
SAVE_INTERVAL=1
MAX_EPOCH=1
CHECKPOINT_SUFFIX=${ADDITIONAL}_${VERSION}_${DATA_PATH}_UF${UPDATE_FREQ}_LR${LR}_WU${WARMUP_UPDATES}_C${LATENT_DIM}_P${PERIOD}

CUDA_VISIBLE_DEVICES=${GPUS} python train.py data/${DATA_PATH}-large-bin \
    --user-dir $USERDIR  \
    --arch $ARCH \
    --pretrained \
    --pretrained-checkpoint $BART_PATH \
    --latent-alpha $LATENT_ALPHA \
    --latent-dim $LATENT_DIM \
    --max-tokens $MAX_TOKENS \
    --task $TASK \
    --period $PERIOD \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --criterion $CRITERION \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --save-interval $SAVE_INTERVAL \
    --checkpoint-suffix $CHECKPOINT_SUFFIX \
    --max-epoch $MAX_EPOCH ;

