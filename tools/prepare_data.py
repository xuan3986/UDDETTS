import argparse
import logging
import glob
import os
from tqdm import tqdm
import random
logger = logging.getLogger()


def main():
    zh_group = {f"{i:04d}" for i in range(1, 11)}
    dev_utts, train_utts = set(), set()
    utt2wav, utt2text, utt2spk, utt2emo, spk2utt_train, spk2utt_dev = {}, {}, {}, {}, {}, {}
    for speaker in os.listdir(args.src_dir):
        if speaker in zh_group:
            continue
        
        data_path = os.path.join(args.src_dir, speaker)

        wavs = list(glob.glob('{}/*/*wav'.format(data_path)))

        if speaker in {"bea", "jenie", "sam"}:
            n = 15 
        elif speaker in {"josh"}:
            n = 5
        elif speaker in {f"{i:04d}" for i in range(11, 21)}:
            n = 5
        elif speaker in {f"Actor_{i:02d}" for i in range(1, 25)}:
            n = 1
        else:
            raise ValueError('Speaker {} not found in groups'.format(speaker))
        

        selected_wavs = random.sample(wavs, n)
        if speaker not in spk2utt_train:
            spk2utt_train[speaker] = []
        if speaker not in spk2utt_dev:
            spk2utt_dev[speaker] = []
        
        
        for wav in tqdm(wavs):
            txt = wav.replace('.wav', '.normalized_emo.txt')
            if not os.path.exists(txt):
                logger.warning('{} do not exsist'.format(txt))
                continue
            with open(txt) as f:
                content = ''.join(l.replace('\n', '') for l in f.readline())
                label = content.split("<endofprompt>")[0]
                content = content.split("<endofprompt>")[1]
                if label == "":
                    logger.warning('{} do not exsist label'.format(wav))
                if content == "":
                    logger.warning('{} do not exsist text'.format(wav))
                    
                    
            utt = os.path.basename(wav).replace('.wav', '')
            utt2wav[utt] = wav
            utt2text[utt] = content
            utt2emo[utt] = label
            utt2spk[utt] = speaker

            if wav in selected_wavs:
                dev_utts.add(utt)
                spk2utt_dev[speaker].append(utt)
            else:
                train_utts.add(utt)
                spk2utt_train[speaker].append(utt)
    
    
    for subset_name in ['dev', 'train']:
        if subset_name == 'dev':
            utts = dev_utts
            spk2utt = spk2utt_dev
        else:
            utts = train_utts
            spk2utt = spk2utt_train

        subset_dir = f"{args.des_dir}/{subset_name}"
        os.makedirs(subset_dir, exist_ok=True)
        with open(f"{subset_dir}/wav.scp", 'w') as f:
            for utt in utts:
                f.write(f"{utt} {utt2wav[utt]}\n")
        with open(f"{subset_dir}/text", 'w') as f:
            for utt in utts:
                f.write(f"{utt} {utt2text[utt]}\n")
        with open(f"{subset_dir}/utt2spk", 'w') as f:
            for utt in utts:
                f.write(f"{utt} {utt2spk[utt]}\n")
        with open(f"{subset_dir}/utt2emo", 'w') as f:
            for utt in utts:
                f.write(f"{utt} {utt2emo[utt]}\n")
        with open(f"{subset_dir}/spk2utt", 'w') as f:
            for k, v in spk2utt.items():
                f.write('{} {}\n'.format(k, ' '.join(v)))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    args = parser.parse_args()
    main()
