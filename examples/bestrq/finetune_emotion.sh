#!/bin/bash
export PYTHONPATH=/llm/nankai/zhoujiaming_space/project/wenet_downstream
export NCCL_DEBUG=INFO

exp_tag="20220506_u2pp_conformer_exp"

echo "begin training"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export master=`scontrol show hostname $SLURM_NODELIST | head -n1`
nodes=`scontrol show hostname $SLURM_NODELIST | wc -l`
node_rank=`echo $SLURM_PROCID`
master_addr=`python -c 'import socket; import os  ; print(socket.gethostbyname(os.environ["master"]))'` ;
echo $ip
echo master ${master}
echo nodes  ${nodes}
echo master_addr ${master_addr}
echo node_rank ${node_rank}

train_path=$PYTHONPATH/wenet/bin/train_spring_emotion.py
train_shards=data/data.list
cv_shards=data/data.list

model=exp/$exp_tag/ckpts/
tensorboard=exp/$exp_tag

config=exp/$exp_tag/train.yaml

mkdir -p $model
mkdir -p $tensorboard
cmvn=exp/$exp_tag/global_cmvn
checkpoint=exp/$exp_tag/final.pt

python ${train_path} --gpu 1 \
           --data_type raw \
           --train_data  ${train_shards} \
           --model_dir ${model} \
           --symbol_table data/dict \
           --ddp.init_method "env://" \
           --cv_data ${cv_shards} \
           --config ${config} \
           --num_workers 4 \
           --tensorboard_dir  ${tensorboard} \
           --cmvn  $cmvn \
           --checkpoint $checkpoint


