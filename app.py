import gradio as gr
import torch
import numpy as np
import sys
from vinorm import TTSnorm
from utils_audio import convert_to_wav
sys.path.append("vits")
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write
import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


from resemblyzer import preprocess_wav, VoiceEncoder


device = "cpu"

def get_text(texts, hps):
    text_norm_list = []
    for text in texts.split(","):
          chunk_strings = []
          chunk_len = 30
          for i in range(0, len(text.split()), chunk_len):
               chunk = " ".join(text.split()[i:i+chunk_len])
               chunk_strings.append(chunk)
          for chunk_string in chunk_strings:
               text_norm = text_to_sequence(chunk_string, hps.data.text_cleaners)
               if hps.data.add_blank:
                    text_norm = commons.intersperse(text_norm, 0)
               text_norm_list.append(torch.LongTensor(text_norm))
    return text_norm_list

def get_speaker_embedding(path):
     encoder = VoiceEncoder(device='cpu')
     path = convert_to_wav(path)
     wav = preprocess_wav(path)
     embed = encoder.embed_utterance(wav)
     return embed
    
class VoiceClone():
     def __init__(self, checkpoint_path):
          hps = utils.get_hparams_from_file("./vits/configs/vivos.json")
          self.net_g = SynthesizerTrn(
          len(symbols),
          hps.data.filter_length // 2 + 1,
          hps.train.segment_size // hps.data.hop_length,
          n_speakers=hps.data.n_speakers,
          **hps.model).to(device)
          _ = self.net_g.eval()

          _ = utils.load_checkpoint(checkpoint_path, self.net_g, None)

          self.hps = hps

     def infer(self, text, ref_audio):
          text_norm =TTSnorm(text)
          stn_tst_list = get_text(text_norm, self.hps)
          with torch.no_grad():
               audios = []
               for stn_tst in stn_tst_list:
                    x_tst = stn_tst.to(device).unsqueeze(0)
                    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)

                    speaker_embedding = get_speaker_embedding(ref_audio)
                    speaker_embedding = torch.FloatTensor(torch.from_numpy(speaker_embedding)).unsqueeze(0).to(device)

                    audio = self.net_g.infer(x_tst, x_tst_lengths, speaker_embedding=speaker_embedding, noise_scale=.667, noise_scale_w=0.8, length_scale=1)

                    audio = audio[0][0,0].data.cpu().float().numpy()
                    audios.append(audio)
                    print(audio.shape)

               audios = np.concatenate(audios, axis=0)
               write(ref_audio.replace(".wav", "_clone.wav"), 22050, audios)
          return ref_audio.replace(".wav", "_clone.wav"), text_norm

object = VoiceClone("vits/logs/vivos/G_7700000.pth")

def clonevoice(text: str, speaker_wav, file_upload, language: str):

     speaker_source = ""
     if speaker_wav is not None:
          speaker_source = speaker_wav
     elif file_upload is not None:
          speaker_source = file_upload
     else:
          speaker_source = "vits/audio/sontung.wav"

     print(speaker_source)

     outfile, text_norm = object.infer(text, speaker_source)
     
     return [outfile, text_norm]

inputs = [gr.Textbox(label="Input", value="muốn ngồi ở một vị trí không ai ngồi được thì phải chịu cảm giác không ai chịu được", max_lines=3),
          gr.Audio(Lable="Speaker Wav", source="microphone", type="filepath"), 
          gr.Audio(Lable="Speaker Wav", source="upload", type="filepath"), 
          gr.Radio(label="Language", choices=["Vietnamese"], value="en")]
outputs = [gr.Audio(label="Output"), gr.TextArea()]

demo = gr.Interface(fn=clonevoice, inputs=inputs, outputs=outputs)

demo.launch(debug=True)