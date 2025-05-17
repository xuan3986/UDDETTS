import os
import numpy as np
import argparse
import logging

def main():
    for mode in ["dev", "train"]:
        out_dir = os.path.join(args.des_dir, mode)
        os.makedirs(out_dir, exist_ok=True)
        for file in ["wav.scp", "text", "utt2ADV", "utt2emo", "utt2spk", "spk2utt"]:
            output_file = os.path.join(out_dir, file)
            combined_lines = []
            for dir in args.src_dirs:
                src_dir = os.path.join(dir, mode)
                if not os.path.exists(src_dir):
                    logging.warning(f"Dir {src_dir} does not exist.") 
                    continue
                src_file = os.path.join(src_dir, file)
                if os.path.exists(src_file):
                    with open(src_file, 'r') as f1:
                        combined_lines += f1.readlines()
                else:
                    raise ValueError(f"File {src_file} does not exist.") 
            np.random.seed(10)
            np.random.shuffle(combined_lines)
            with open(output_file, 'w') as f:
                f.writelines(combined_lines)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dirs', nargs='+', type=str)
    parser.add_argument('--des_dir', type=str)
    args = parser.parse_args()
    main()
