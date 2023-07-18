#!/bin/bash
export PYTHONPATH=/mnt/petrelfs/zhoudinghao/work/thzhang/wenet/
export NCCL_DEBUG=INFO

exp_tag=train_ln
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
# export CUDA_VISIBLE_DEVICES="0,1,2,3"

train_path=$PYTHONPATH/wenet/bin/train_spring.py
train_shards=data/aishell/aishell_train/data_list.txt
cv_shards=data/aishell/aishell_dev/data_list.txt

model=exp/$exp_tag/ckpts/
tensorboard=exp/$exp_tag

config=aishell.yaml

mkdir -p $model
mkdir -p $tensorboard
cmvn=data/aishell/global_cmvn
checkpoint=pre-train.pt

time python launch.py  --nproc_per_node=4 --master_port=52071 \
           --nnodes=$nodes \
           --master_addr $master_addr \
           --node_rank=$node_rank \
           ${train_path} --gpu 1 \
           --data_type shard \
           --train_data  ${train_shards} \
           --model_dir ${model} \
           --symbol_table data/aishell/lang_char.txt \
           --ddp.init_method "env://" \
           --cv_data ${cv_shards} \
           --config ${config} \
           --num_workers 4 \
           --tensorboard_dir  ${tensorboard} \
            --cmvn  $cmvn \
        #    --checkpoint $checkpoint


        #    $cmvn_opts \
                    #   --checkpoint $checkpoint \
            # --checkpoint $checkpoint \
        #    --checkpoint $checkpoint


