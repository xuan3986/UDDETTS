## Datasets
Due to the dataset license requirements, this code does not provide source datasets and cleaning scripts for the time being. You need to download the dataset yourself and build a dataset directory structure as follows:
```
    dataset
    |- EMOTTSDB
        |- bea
            | anger
                | bea_anger_1-28_0001.wav
                | bea_anger_1-28_0001.normalized.txt
                ...
            ...
            | neutral
                | bea_neutral_1-28_0001.wav
                | bea_neutral_1-28_0001.normalized.txt
        ...
        |- sam
        | 0001
        ...
        | 0020
        | Actor_01
        ...
        | Actor_24
    |- IEMOCAP
        |- train
            |- wav.scp
            |- text
            |- spk2utt
            |- utt2spk
            |- utt2emo
            |- utt2ADV
        |- dev
            |- wav.scp
            |- text
            |- spk2utt
            |- utt2spk
            |- utt2emo
            |- utt2ADV
    |- MSP
        |- train
        |- dev
    |- MELD
        |- train
        |- dev
    | Other datasets
```
## Pretrained models
The `pretrained_models` include `campplus.onnx`, `speech_tokenizer.onnx`, `llm.pt`, `flow.pt`, and `hifi.pt`.
Due to the large number of parameters,
these models will be released soon on [Huggingface](https://huggingface.co/) soon.

## Third party modules
Matcha-TTS and roberta are third_party modules. Please check `third_party` directory. 

If there is no `Matcha-TTS`, execute `git clone git@github.com:shivammehta25/Matcha-TTS.git` in `third_party` directory.

If there is no `roberta`, please download from [roberta-base](https://huggingface.co/FacebookAI/roberta-base/tree/main) to `third_party` directory.

