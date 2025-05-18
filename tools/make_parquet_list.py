import argparse
import logging
import os
import json
from tqdm import tqdm
import pandas as pd
import multiprocessing
import time
import torch

emo_dict = {
    "other": 0,
    "unknown": 0,
    "sad": 1,
    "angry": 2,
    "frustrated": 3,
    "disgust": 4,
    "contempt": 4,
    "fearful": 5,
    "sleepiness": 6,
    "calm": 6,
    "neutral": 7,
    "surprise": 8,
    "happy": 9,
    "amused": 9
}
def job(utt_list, parquet_file, utt2parquet_file, spk2parquet_file):
    try:
        start_time = time.time()
        data_list = []
        for utt in tqdm(utt_list):
            data = open(utt2wav[utt], 'rb').read()
            data_list.append(data)
        wav_list = [utt2wav[utt] for utt in utt_list]
        text_list = [utt2text[utt] for utt in utt_list]
        spk_list = [utt2spk[utt] for utt in utt_list]
        emo_list = [utt2emo[utt] for utt in utt_list]
        uttembedding_list = [utt2embedding[utt] for utt in utt_list]
        spkembedding_list = [spk2embedding[utt2spk[utt]] for utt in utt_list]
        speech_token_list = [utt2speech_token[utt] for utt in utt_list]
        ADV_list = [utt2ADV[utt] for utt in utt_list]
        # save in parquet,utt2parquet_file,spk2parquet_file
        df = pd.DataFrame()
        df['utt'] = utt_list
        df['wav'] = wav_list
        df['audio_data'] = data_list
        df['text'] = text_list
        df['spk'] = spk_list
        df['utt_embedding'] = uttembedding_list
        df['spk_embedding'] = spkembedding_list
        df['speech_token'] = speech_token_list
        df['emo_id'] = emo_list
        df['ADV'] = ADV_list
        df.to_parquet(parquet_file)
        with open(utt2parquet_file, 'w') as f:
            json.dump({k: parquet_file for k in utt_list}, f, ensure_ascii=False, indent=2)
        with open(spk2parquet_file, 'w') as f:
            json.dump({k: parquet_file for k in list(set(spk_list))}, f, ensure_ascii=False, indent=2)
        logging.info('spend time {}'.format(time.time() - start_time))
    except Exception as e:
        logging.error('Error in job: {}'.format(e))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet',
                        type=int,
                        default=1000,
                        help='num utts per parquet')
    parser.add_argument('--num_processes',
                        type=int,
                        default=1,
                        help='num processes for make parquets')
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    args = parser.parse_args()

    utt2wav, utt2text, utt2spk, utt2emo, utt2ADV = {}, {}, {}, {}, {}
    with open('{}/wav.scp'.format(args.src_dir)) as f:
        for l in f:
            l = l.strip()
            if not l: continue # 处理空行
            l = l.split()
            utt2wav[l[0]] = l[1]
    with open('{}/text'.format(args.src_dir)) as f:
        for l in f:
            l = l.strip()
            if not l: continue
            l = l.split()
            utt2text[l[0]] = ' '.join(l[1:])
    with open('{}/utt2spk'.format(args.src_dir)) as f:
        for l in f:
            l = l.strip()
            if not l: continue
            l = l.split()
            utt2spk[l[0]] = l[1]
    with open('{}/utt2emo'.format(args.src_dir)) as f:
        for l in f:
            l = l.strip()
            if not l: continue
            l = l.split()
            utt2emo[l[0]] = int(emo_dict[l[1]])
    with open('{}/utt2ADV'.format(args.src_dir)) as f:
        for l in f:
            l = l.strip()
            if not l: continue
            l = l.split(' ', 1)
            utt2ADV[l[0]] = eval(l[1])
    utt2embedding = torch.load('{}/utt2embedding.pt'.format(args.src_dir))
    spk2embedding = torch.load('{}/spk2embedding.pt'.format(args.src_dir))
    utt2speech_token = torch.load('{}/utt2speech_token.pt'.format(args.src_dir))
    utts = list(utt2wav.keys())

    # Using process pool to speedup
    pool = multiprocessing.Pool(processes=args.num_processes)
    parquet_list, utt2parquet_list, spk2parquet_list = [], [], []
    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        parquet_file = os.path.join(args.des_dir, 'parquet_{:09d}.tar'.format(i))
        utt2parquet_file = os.path.join(args.des_dir, 'utt2parquet_{:09d}.json'.format(i))
        spk2parquet_file = os.path.join(args.des_dir, 'spk2parquet_{:09d}.json'.format(i))
        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        spk2parquet_list.append(spk2parquet_file)
        pool.apply_async(job, (utts[j: j + args.num_utts_per_parquet], parquet_file, utt2parquet_file, spk2parquet_file))
    pool.close()
    pool.join()

    with open('{}/data.list'.format(args.des_dir), 'w', encoding='utf8') as f1, \
            open('{}/utt2data.list'.format(args.des_dir), 'w', encoding='utf8') as f2, \
            open('{}/spk2data.list'.format(args.des_dir), 'w', encoding='utf8') as f3:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
        for name in spk2parquet_list:
            f3.write(name + '\n')
