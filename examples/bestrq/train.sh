#!/bin/bash
export PYTHONPATH=/mnt/petrelfs/zhoudinghao/work/thzhang/wenet/
export NCCL_DEBUG=INFO

# exp_tag=bestrq_all_global_cmvn_new_for_new_nospec_sp_noam_long
# exp_tag=all_cmvn_600m_new_test_continue
# exp_tag=douyin_tts_small
exp_tag=librispeech_cmvn_pro_noln_dynamic_6
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
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

train_path=$PYTHONPATH/wenet/bin/train_spring.py
train_shards=data/librispeech_shard/data_list.txt
cv_shards=data/zh_en_shuf_20230714/data_list.txt

model=exp/$exp_tag/ckpts/
tensorboard=exp/$exp_tag/tensorboard/

# config=bestrq.yaml
config=conf/bestrq.yaml
# config='/mnt/petrelfs/share_data/zhoudinghao/.tmp/big/train.yaml'

mkdir -p $model
mkdir -p $tensorboard
cmvn=data/librispeech_global_cmvn
checkpoint='exp/librispeech_cmvn_pro_noln_dynamic_6/ckpts/20000.pt'

time python launch.py  --nproc_per_node=8 --master_port=52019 \
           --nnodes=$nodes \
           --master_addr $master_addr \
           --node_rank=$node_rank \
           ${train_path} --gpu 1 \
           --data_type shard \
           --train_data  ${train_shards} \
           --model_dir ${model} \
           --symbol_table data/librispeech_shard/librispeech_dict \
           --ddp.init_method "env://" \
           --cv_data ${cv_shards} \
           --config ${config} \
           --num_workers 4 \
           --tensorboard_dir  ${tensorboard} \
           --cmvn  $cmvn \
           $cmvn_opts \
           --checkpoint $checkpoint
