#!/bin/bash
. ./path.sh || exit 1;

stage=7
stop_stage=7
# EMOTTSDB is D_{E,L}, includes Preprocessed emovdb RAVDESS ESD
data_dir=/home/jxliu/workspace/dataset/EMOTTSDB
pretrained_model_dir=../../pretrained_models

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation for D_{E,L}, prepare wav.scp/text/utt2spk/utt2emo/spk2utt"
    mkdir -p data
    python tools/prepare_data.py \
      --src_dir $data_dir \
      --des_dir data/EMOTTSDB
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "ADV nonlinear quantization"
    python tools/ADV_binning.py \
      --data_path data/Total \
      --pic_path tools/pictures
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Sum up all data to Total, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2emo/utt2ADV"
  for x in dev; do
    python tools/total_dataset.py \
      --src_dirs data/EMOTTSDB data/IEMOCAP data/MELD data/MSP \
      --des_dir data/Total
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt"
    python tools/extract_embedding.py \
      --dir data/Total \
      --onnx_path $pretrained_model_dir/campplus.onnx
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt"
  for x in dev train; do
    python tools/extract_speech_token.py \
      --dir data/Total/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer.onnx
  done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2emo/utt2ADV/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in dev train; do
    mkdir -p data/Total/$x/parquet
    python tools/make_parquet_list.py \
      --num_utts_per_parquet 2000 \
      --num_processes 20 \
      --src_dir data/Total/$x \
      --des_dir data/Total/$x/parquet
  done
fi


# inference
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Run inference. Please make sure utt in tts_text is in prompt_data"
  for A in 7; do
    for D in 7; do
      for V in 7; do
        python UDDETTS/bin/inference.py \
          --gpu 3 \
          --config conf/UDDETTS.yaml \
          --prompt_data data/Total/dev/parquet/data.list \
          --prompt_utt2data data/Total/dev/parquet/utt2data.list \
          --tts_text `pwd`/tts_text.json \
          --llm_model exp/llm/ADV/best_40000.pt\
          --flow_model exp/flow/ADV/best_73000.pt \
          --hifigan_model $pretrained_model_dir/hift.pt \
          --result_dir `pwd`/exp/result/control \
          --ADV_list $A $D $V \
          --use_ADV
      done
    done
  done
fi


# train llm and flow and hifigan
export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
dist_backend="gloo" # "nccl" or "gloo"
num_workers=10
prefetch=100
train_engine=torch_ddp
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage.json if necessary"
  fi

  for model in llm flow; do
    python -m torch.distributed.run --nproc_per_node=$num_gpus \
      UDDETTS/bin/train.py \
      --train_engine $train_engine \
      --config conf/UDDETTS.yaml \
      --train_data data/Total/train/parquet/data.list \
      --cv_data data/Total/dev/parquet/data.list \
      --model $model \
      --checkpoint $pretrained_model_dir/${model}.pt \
      --model_dir `pwd`/exp/$model \
      --tensorboard_dir `pwd`/tensorboard/$model \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ./conf/ds_stage.json \
      --deepspeed.save_states model+optimizer \
      --timeout 120
  done
fi
# --checkpoint exp/${model} \