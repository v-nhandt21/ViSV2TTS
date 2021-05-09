import sys
sys.path.append('/home/ubuntu/Real-Time-Voice-Cloning/encoder')
# from encoder import inference as encoder
from hparams import hparams
from audio import melspectrogram
import librosa
import numpy as np      

import glob
emb_list = glob.glob("/vinai/nhandt21/VoiceClone/embed/*.npy")
emb_list = [e.replace("/vinai/nhandt21/VoiceClone/embed/","").split(".")[0] for e in emb_list]

num_no_emb =0 
with open("/vinai/nhandt21/VoiceClone/train.txt","w+",encoding="utf-8") as fw:
    with open("/vinai/nhandt21/VoiceClone/bigtext.txt","r",encoding="utf-8") as f:
        sample_rate = 22050
        lines = f.read().splitlines()
        
        for line in lines:
            wav_fpath , script = line.split("|")

            wav_fpath = "/vinai/nhandt21/VoiceClone/"+wav_fpath
            file_name = wav_fpath.replace("/vinai/nhandt21/VoiceClone/vlsp2020_train_set_02/","").replace(".wav","")

            mel_fpath = "/vinai/nhandt21/VoiceClone/mels/"+file_name+".npy"
            embed_fpath = "/vinai/nhandt21/VoiceClone/embed/"+file_name+".npy"
            
            if file_name not in emb_list:
                num_no_emb = num_no_emb+1
                print(file_name)
                print("=============")
                continue
            # try:

            wav, sr = librosa.load(str(wav_fpath))#, sample_rate)
            if sr != 22050:
                print(sr)

            rescale = False
            # if rescale:
            #     wav = wav / np.abs(wav).max() * hparams.rescaling_max
                
            trim_silence = False
            # if trim_silence:
            #     wav = encoder.preprocess_wav(wav, normalize=False, trim_silence=True)
            
            # Skip utterances that are too short
            if len(wav) < 2 * sample_rate:
                continue
            
            # Compute the mel spectrogram
            mel_spectrogram = melspectrogram(wav, hparams).astype(np.float32)
            mel_frames = mel_spectrogram.shape[1]
            
            # Skip utterances that are too long
            # if mel_frames > 900:
            #     print("Audio is too long")
            #     return None
            
            # Write the spectrogram, embed and audio to disk
            np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
            fw.write(wav_fpath+"|"+mel_fpath+"|"+embed_fpath+"|nhan|tri|"+script+"\n")
            # except:
            #     print("Do not save this file")
            #     continue
print(num_no_emb)