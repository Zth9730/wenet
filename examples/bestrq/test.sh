#!/bin/bash
export PYTHONPATH=/mnt/petrelfs/zhoudinghao/work/thzhang/wenet/
export NCCL_DEBUG=INFO

exp_tag=fintune_ctc_librispeech_960_dh_big
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

test_path=$PYTHONPATH/wenet/bin/recognize.py

dict=data/librispeech_shard/librispeech_dict
# dict=data/lang_chartrain_960_unigram5000.vocab
bpemodel=data/lang_char/train_960_unigram5000

model=exp/$exp_tag/ckpts/

average_checkpoint=true
average_num=10
# decode_checkpoint=${model}/390.pt
decode_checkpoint=${model}/avg_${average_num}.pt
decode_modes="ctc_greedy_search"
# test_set="test_other"
test_set="test_other"


# Test model, please specify the model you want to test by --checkpoint
if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$model/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python ../../wenet/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $model  \
        --num ${average_num} \
        --val_best
fi
#   # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
decoding_chunk_size=
ctc_weight=1.0
reverse_weight=0.0

for mode in ${decode_modes}; do
{
    for sets in ${test_set}; do
    {
        test_shards=data/librispeech_shard/librispeech_${sets}_shard/data_list.txt
        test_dir=${model}/${sets}_${mode}
        mkdir -p $test_dir
        python ${test_path} --gpu 1 \
            --mode ${mode} \
            --config ${model}/train.yaml \
            --data_type shard \
            --test_data ${test_shards} \
            --checkpoint $decode_checkpoint \
            --beam_size 1 \
            --batch_size 80 \
            --penalty 0.0 \
            --dict ${dict} \
            --bpe_model ${bpemodel}.model \
            --ctc_weight $ctc_weight \
            --reverse_weight $reverse_weight \
            --result_file $test_dir/text \
            --connect_symbol ''  ## used for phone 
        python ../../tools/compute-wer.py --char=1 --v=1 \
            data/librispeech_shard/${sets}.txt $test_dir/text > $test_dir/wer
        echo 'done'
}
    done
} 
done
