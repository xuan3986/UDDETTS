from __future__ import print_function

import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
import torch
from torch.utils.data import DataLoader
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
from UDDETTS.bin.model import UDDETTS
from UDDETTS.dataset.dataset import Dataset


def get_args():
    parser = argparse.ArgumentParser(description='inference with your model')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--prompt_data', required=True, help='prompt data file')
    parser.add_argument('--prompt_utt2data', required=True, help='prompt data file')
    parser.add_argument('--tts_text', required=True, help='tts input file')
    parser.add_argument('--llm_model', required=True, help='llm model file')
    parser.add_argument('--flow_model', required=True, help='flow model file')
    parser.add_argument('--hifigan_model', required=True, help='hifigan model file')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--use_ADV', 
                        action='store_true', 
                        default=False, 
                        help='Use ADV to control the output emotion')
    parser.add_argument('--ADV_list',
                        nargs='+',
                        type=int,
                        help='if use ADV')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Init cosyvoice models from configs
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f)

    model = UDDETTS(configs['llm'], configs['flow'], configs['hift'], fp16=False)
    model.load(args.llm_model, args.flow_model, args.hifigan_model)

    test_dataset = Dataset(args.prompt_data, data_pipeline=configs['data_pipeline'], mode='inference', shuffle=False, partition=False,
                           tts_file=args.tts_text, prompt_utt2data=args.prompt_utt2data)
    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    del configs
    os.makedirs(args.result_dir, exist_ok=True)
    fn = os.path.join(args.result_dir, 'wav.scp')
    f = open(fn, 'w')
    with torch.no_grad():
        for _, batch in tqdm(enumerate(test_data_loader)):
            utts = batch["utts"]
            assert len(utts) == 1, "inference mode only support batchsize 1"
            tts_index = batch["tts_index"]
            tts_text_token = batch["tts_text_token"].to(device)
            spk_embedding = batch["spk_embedding"].to(device)
            if args.use_ADV:
                ADV_token = torch.tensor(args.ADV_list, dtype=torch.int32)
            else:
                ADV_token = torch.tensor([0, 0, 0], dtype=torch.int32)
            ADV_token = ADV_token.unsqueeze(0).repeat(tts_text_token.shape[0], 1)

            model_input = {
                'text': tts_text_token, 
                'ADV_token': ADV_token ,
                'spk_embedding': spk_embedding,
                'stream': False, 'speed': 1.0
            }

            tts_speeches = []
            for model_output in model.tts(**model_input):
                tts_speeches.append(model_output['tts_speech'])
            tts_speeches = torch.concat(tts_speeches, dim=1)
            tts_key = '{}_{}'.format(utts[0], tts_index[0])
            if args.use_ADV:
                tts_key = '{}-{}-{}'.format(args.ADV_list[0], args.ADV_list[1], args.ADV_list[2])
            tts_fn = os.path.join(args.result_dir, '{}.wav'.format(tts_key))
            torchaudio.save(tts_fn, tts_speeches, sample_rate=22050)
            f.write('{} {}\n'.format(tts_key, tts_fn))
            f.flush()
    f.close()
    logging.info('Result wav.scp saved in {}'.format(fn))


if __name__ == '__main__':
    main()
