#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

hpu=1

nhpus="1"

dlrm_pt_bin="python dlrm_dcn_s_pytorch_hpu.py"

data=random_yj #synthetic
rand_seed=1993

mkdir -p logs/dlrm_dcn

###############################################################
# DLRM_DCN_RM1
mini_batch_size=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
emb_dim=(4 8 16 32 64 128 256 512)
nbatches=101 #500 #100
# bot_mlp="512-256-64"
top_mlp="1024-1024-512-256-1"
dcn_num_layers=3
dcn_low_rank_dim=512
nlookup=10
emb="1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000"

#_args="--mini-batch-size="${mb_size}\
_args=" --num-batches="${nbatches}\
" --data-generation="${data}\
" --arch-mlp-top="${top_mlp}\
" --dcn-num-layers="${dcn_num_layers}\
" --dcn-low-rank-dim="${dcn_low_rank_dim}\
" --arch-embedding-size="${emb}\
" --num-indices-per-lookup="${nlookup}\
" --num-indices-per-lookup-fixed=True"\
" --numpy-rand-seed="${rand_seed}\
" --use-custom-fbgemm"\
" --inference-only"

# HPU Benchmarking
if [ $hpu = 1 ]; then
  echo "--------------------------------------------"
  echo "HPU Benchmarking - running on $nhpus HPUs"
  echo "--------------------------------------------"
  for _ng in $nhpus
  do
    for _emb_dim in "${emb_dim[@]}"
    do
      for _mini_batch_size in "${mini_batch_size[@]}"
      do
        _hpus=$(seq -s, 0 $((_ng-1)))
        hpu_arg="HABANA_VISIBLE_MODULES=$_hpus"
        echo "-------------------"
        echo "Using HPUS: "$_hpus
        echo "-------------------"
        outf="logs/dlrm_dcn/DLRM_DCN_RM1_HPU_PT_END_TO_END_dim${_emb_dim}_mb${_mini_batch_size}.log"
        echo "-------------------------------"
        echo "Running PT (log file: $outf)"
        echo "-------------------------------"
        cmd="$hpu_arg $dlrm_pt_bin --arch-sparse-feature-size $_emb_dim --arch-mlp-bot=512-256-$_emb_dim --mini-batch-size=$_mini_batch_size $_args --test-file-name=dataset/DLRM_DCN_RM1_FBGEMM_random_batchsize$_mini_batch_size.pt --test-load --use-hpu > $outf"
        echo $cmd
        eval $cmd
      done
    done
  done
fi

###############################################################
# DLRM_DCN_RM2
mini_batch_size=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
emb_dim=(4 8 16 32 64 128 256 512)
nbatches=101 #500 #100
# bot_mlp="256-64-64"
top_mlp="128-64-1"
dcn_num_layers=2
dcn_low_rank_dim=64
nlookup=100
emb="1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000"

#_args="--mini-batch-size="${mb_size}\
_args=" --num-batches="${nbatches}\
" --data-generation="${data}\
" --arch-mlp-top="${top_mlp}\
" --dcn-num-layers="${dcn_num_layers}\
" --dcn-low-rank-dim="${dcn_low_rank_dim}\
" --arch-embedding-size="${emb}\
" --num-indices-per-lookup="${nlookup}\
" --num-indices-per-lookup-fixed=True"\
" --numpy-rand-seed="${rand_seed}\
" --use-custom-fbgemm"\
" --inference-only"

# HPU Benchmarking
if [ $hpu = 1 ]; then
  echo "--------------------------------------------"
  echo "HPU Benchmarking - running on $nhpus HPUs"
  echo "--------------------------------------------"
  for _ng in $nhpus
  do
    for _emb_dim in "${emb_dim[@]}"
    do
      for _mini_batch_size in "${mini_batch_size[@]}"
      do
        _hpus=$(seq -s, 0 $((_ng-1)))
        hpu_arg="HABANA_VISIBLE_MODULES=$_hpus"
        echo "-------------------"
        echo "Using HPUS: "$_hpus
        echo "-------------------"
        outf="logs/dlrm_dcn/DLRM_DCN_RM2_HPU_PT_END_TO_END_dim${_emb_dim}_mb${_mini_batch_size}.log"
        echo "-------------------------------"
        echo "Running PT (log file: $outf)"
        echo "-------------------------------"
        cmd="$hpu_arg $dlrm_pt_bin --arch-sparse-feature-size $_emb_dim --arch-mlp-bot=256-64-$_emb_dim --mini-batch-size=$_mini_batch_size $_args --test-file-name=dataset/DLRM_DCN_RM2_FBGEMM_random_batchsize$_mini_batch_size.pt --test-load --use-hpu > $outf"
        echo $cmd
        eval $cmd
      done
    done
  done
fi