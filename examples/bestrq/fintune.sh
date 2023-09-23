#!/bin/bash

export PYTHONPATH=/mnt/petrelfs/zhoudinghao/work/thzhang/wenet/
export PYTHONPATH=/tmp/icefall:$PYTHONPATH
export NCCL_DEBUG=INFO

exp_tag=train_librispeech_ctc_bpe2
echo "begin training"
export master=`scontrol show hostname $SLURM_NODELIST | head -n1`
nodes=`scontrol show hostname $SLURM_NODELIST | wc -l`
node_rank=`echo $SLURM_PROCID`
master_addr=`python -c 'import socket; import os  ; print(socket.gethostbyname(os.environ["master"]))'` ;
echo $ip
echo master ${master}
echo nodes  ${nodes}
echo master_addr ${master_addr}
echo node_rank ${node_rank}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

train_path=/mnt/petrelfs/zhoudinghao/work/thzhang/wenet/wenet/bin/train_fintune.py
train_shards=data/librispeech_shard/librispeech_train_100/data_list.txt
cv_shards=data/librispeech_shard/librispeech_dev_clean_shard/data_list.txt

model=exp/$exp_tag/ckpts/
tensorboard=exp/$exp_tag
dict=data/lang_char/train_960_unigram5000_units.txt
bpemodel=data/lang_char/train_960_unigram5000
# config=aishell.yaml
config=conf/fintune_ctc.yaml


mkdir -p $model
mkdir -p $tensorboard
cmvn=data/librispeech_global_cmvn
checkpoint='124000.pt'
# checkpoint=exp/fintune_ctc_librispeech_960/ckpts/139.pt
# config=exp/fintune_ctc_librispeech_960/ckpts/train.yaml

time python launch.py  --nproc_per_node=8 --master_port=52079 \
        --nnodes=$nodes \
        --master_addr $master_addr \
        --node_rank=$node_rank \
        ${train_path} --gpu 1 \
        --data_type shard \
        --train_data  ${train_shards} \
        --model_dir ${model} \
        --symbol_table ${dict} \
        --bpe_model ${bpemodel}.model \
        --ddp.init_method "env://" \
        --cv_data ${cv_shards} \
        --config ${config} \
        --num_workers 4 \
        --tensorboard_dir  ${tensorboard} \
        --cmvn  $cmvn 
        --checkpoint $checkpoint 
        # --cmvn  $cmvn \
        # $cmvn_opts



        #    $cmvn_opts \
                    #   --checkpoint $checkpoint \
            # --checkpoint $checkpoint \
        #    --checkpoint $checkpoint


