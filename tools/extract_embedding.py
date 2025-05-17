import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm

def single_job(utt):
    audio, sample_rate = torchaudio.load(utt2wav[utt])
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    feat = kaldi.fbank(audio,
                       num_mel_bins=80,
                       dither=0,
                       sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
    return utt, embedding


def main(args):
    all_task = [executor.submit(single_job, utt) for utt in utt2wav.keys()]
    utt2embedding_dev, utt2embedding_train, spk2embedding = {}, {}, {}
    for future in tqdm(as_completed(all_task)):
        utt, embedding = future.result()
        spk = utt2spk[utt]
        dir = utt2dir[utt]
        if dir == "dev":
            utt2embedding_dev[utt] = embedding
        elif dir == "train":
            utt2embedding_train[utt] = embedding
        else:
            raise ValueError("{} is not in dev or train".format(dir))

        if spk not in spk2embedding:
            spk2embedding[spk] = []
        spk2embedding[spk].append(embedding)
    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()

    torch.save(utt2embedding_dev, "{}/dev/utt2embedding.pt".format(args.dir))
    torch.save(utt2embedding_train, "{}/train/utt2embedding.pt".format(args.dir))
    torch.save(spk2embedding, "{}/dev/spk2embedding.pt".format(args.dir))
    torch.save(spk2embedding, "{}/train/spk2embedding.pt".format(args.dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--onnx_path", type=str)
    parser.add_argument("--num_thread", type=int, default=24)
    args = parser.parse_args()

    utt2wav, utt2spk, utt2emo, utt2dir = {}, {}, {}, {}
    for subdir in ["dev", "train"]:
        with open('{}/{}/wav.scp'.format(args.dir, subdir)) as f:
            for l in f:
                l = l.replace('\n', '').split()
                utt2wav[l[0]] = l[1]
                utt2dir[l[0]] = subdir
        with open('{}/{}/utt2spk'.format(args.dir, subdir)) as f:
            for l in f:
                l = l.replace('\n', '').split()
                utt2spk[l[0]] = l[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
    executor = ThreadPoolExecutor(max_workers=args.num_thread)

    main(args)
