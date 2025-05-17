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


## Module 'matcha' and 
Matcha-TTS is a third_party module. Please check `third_party` directory. If there is no `Matcha-TTS`, execute `git clone git@github.com:shivammehta25/Matcha-TTS.git` in `third_party` directory.