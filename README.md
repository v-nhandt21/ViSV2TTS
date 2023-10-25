# Vietnamese Voice Clone

## Data Preparation

***If you use custom data***

- Config your custom data follow this format:

     - Create folder: DATA

     - Subfolder: DATA/wavs -> which contain <audio_id>.wav files inside

     - DATA/train.txt and DATA/val.txt: with format each line follow format: <audio_id><space>transcript

- If you dont have transcript, please check wav2vec inference script

***If you try with VIVOS***

```
wget http://ailab.hcmus.edu.vn/assets/vivos.tar.gz
tar xzf vivos.tar.gz
```

```
mkdir -p DATA/wavs
scp -v vivos/*/waves/*/*.wav DATA/wavs
```

```
cat vivos/test/prompts.txt > DATA/val.txt
cat vivos/test/prompts.txt > DATA/train.txt
cat vivos/train/prompts.txt >> DATA/train.txt
```

## Install environment

```
conda create -y -n viclone python=3.8
conda activate viclone
conda install cudatoolkit=11.3.1 cudnn=8.2.1
```

```
python -m pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r requirements.txt
```

```
cd vits/monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
```

## Process data

```
python Step1_data_processing.py
```

## Extract feature

```
python Step2_extract_feature.py
```

## Train model

```
python train_ms.py -c configs/vivos.json -m vivos 
```

## Demo

```python app.py```

Then check port: http://127.0.0.1:7860/