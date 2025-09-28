# UDDETTS: Unifying Discrete and Dimensional Emotions for Controllable Emotional Text-to-Speech

[Demo page](https://anonymous.4open.science/w/UDDETTS/);
[Paper](https://openreview.net/pdf?id=DuPYSaCiep);
[Hugging Face]();

## Highlight🔥
**UDDETTS** has been released! Compared to the previous version, some modules have been updated, 
and it is trained on over ten thousand hours of speech data, making the model more stable and robust.

## Install
- Clone the repo:
    ``` sh
    git clone --recursive https://github.com/xuan3986/UDDETTS.git
    cd UDDETTS
    git submodule update --init --recursive
    ```

- Create Conda env:

    ``` sh
    conda create -n UDDETTS -y python=3.8.20
    conda activate UDDETTS
    pip install -r requirements.txt
    ```
- Model download:
  
To ensure anonymity, pre-trained models trained on large-scale emotional speech datasets will be released on the open-source platform after review. Thank you for your patience.

## Usage
[1] Download English emotional speech datasets:
1. [MSP-Podcast](https://lab-msp.com/MSP/MSP-Podcast.html)
2. [IEMOCAP](https://sail.usc.edu/iemocap/)
3. [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)
4. [Expresso](https://huggingface.co/datasets/ylacombe/expresso)
5. [MELD](https://github.com/declare-lab/MELD)
6. [EmoTale](https://github.com/snehadas/EmoTale)
7. [EU-Emotion](https://pmc.ncbi.nlm.nih.gov/articles/PMC6478635)
8. [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data)
9. [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
10. [EmoV-DB](https://www.openslr.org/115/)
11. [MEAD](https://github.com/uniBruce/Mead)
12. [RAVDESS](https://github.com/tuncayka/speech_emotion)

[2] Preprocess, since each dataset follows a different organization format, we handle them individually. We provide partial code and processed data samples as references.

[3] Extract features, including ADV bins, speaker embedding, speech tokens, parquet list...
(Stage 1-5)


[4] Train (Stage 7)

``` sh
    cd examples/exp1
    conda activate UDDETTS
    bash run.sh
```

[5] Inference (Stage 6)

A D V in range(1, 14), you can use a text from the test_examples.

``` sh
    cd examples/exp1
    conda activate UDDETTS
    bash run.sh
```

## Roadmap

- [x] 2025/09

    - [x] We modify the code and upload some lightweight model files and data samples for demonstration.

- [x] 2025/05

    - [x] Release the core architecture and base code of UDDETTS

## License
The UDDETTS model can be used for non-commercial purposes, see [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). 
The source code in this GitHub repository 
is released under the following [license](./LICENSE).

## Acknowledge
1. [CosyVoice](https://github.com/FunAudioLLM/CosyVoice).
2. [whisper](https://github.com/openai/whisper).
3. [Matcha](https://github.com/shivammehta25/Matcha-TTS).
4. [roberta-base](https://huggingface.co/FacebookAI/roberta-base).
5. [3D-Speaker](https://github.com/modelscope/3D-Speaker)
6. [hifi-gan](https://github.com/jik876/hifi-gan).


## Disclaimer
The content provided above is for academic purposes only. Some content is sourced from the internet. If any content infringes on your rights, please contact us to request its removal.