import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import torch
from tqdm import tqdm
import onnxruntime
import numpy as np
import torchaudio
import whisper

def single_job(utt):
    audio, sample_rate = torchaudio.load(utt2wav[utt])
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    if audio.shape[0] != 1:
        logging.warning('Audio {} is not a single channel, shape is {}'.format(utt, audio.shape))
        audio = audio.mean(dim=0, keepdim=True)
    if audio.shape[1] / 16000 > 30:
        logging.warning('Do not support extract speech token for audio longer than 30s')
        speech_token = []
    else:
        feat = whisper.log_mel_spectrogram(audio, n_mels=128)
        try:
            speech_token = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                                                ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        except Exception as e:
            logging.warning('extract speech token failed for {}'.format(utt))
    return utt, speech_token


def main(args):
    all_task = [executor.submit(single_job, utt) for utt in utt2wav.keys()]
    utt2speech_token = {}
    for future in tqdm(as_completed(all_task)):
        utt, speech_token = future.result()
        utt2speech_token[utt] = speech_token
    torch.save(utt2speech_token, '{}/utt2speech_token.pt'.format(args.dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--onnx_path", type=str)
    parser.add_argument("--num_thread", type=int, default=30)
    args = parser.parse_args()

    utt2wav = {}
    with open('{}/wav.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
    executor = ThreadPoolExecutor(max_workers=args.num_thread)

    main(args)
