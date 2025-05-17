"""Predicting the ADV value of audio, But the error is a bit large"""
import argparse
from tqdm import tqdm
import librosa
import audonnx
from concurrent.futures import ThreadPoolExecutor, as_completed
sampling_rate = 16000

def single_job(utt):
    signal, _ = librosa.load(utt2wav[utt], sr=sampling_rate)
    logits = model(signal, sampling_rate)['logits'][0]
    arousal, dominance, valence = logits[0], logits[1], logits[2]
    arousal = max(0.0, min(1.0, arousal)) * 6.0 + 1.0
    dominance = max(0.0, min(1.0, dominance)) * 6.0 + 1.0
    valence = max(0.0, min(1.0, valence)) * 6.0 + 1.0
    ADV = [arousal, dominance, valence]
    return utt, ADV

def main(args):
    all_task = [executor.submit(single_job, utt) for utt in utt2wav.keys()]
    utt2ADV = {}
    for future in tqdm(as_completed(all_task)):
        utt, ADV = future.result()
        utt2ADV[utt] = ADV
        
    with open('{}/utt2ADV'.format(args.dir), 'w') as f:
        for k, v in utt2ADV.items():
            f.write('{} {}\n'.format(k, v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument("--num_thread", type=int, default=32)
    args = parser.parse_args()
    
    utt2wav, utt2spk = {}, {}
    with open('{}/wav.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    
    model_root = 'tools/speechadvpredictor'
    model = audonnx.load(model_root)
    print(model)
    executor = ThreadPoolExecutor(max_workers=args.num_thread)
    
    main(args)
