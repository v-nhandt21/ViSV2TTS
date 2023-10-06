import torch
import numpy as np
import sys
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write
import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

sys.path.append("../")
from resemblyzer import preprocess_wav, VoiceEncoder


device = "cpu"

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_speaker_embedding(path):
     encoder = VoiceEncoder(device='cpu')
     wav = preprocess_wav(path)
     embed = encoder.embed_utterance(wav)
     return embed
    
class VoiceClone():
     def __init__(self, checkpoint_path):
          hps = utils.get_hparams_from_file("./configs/vivos.json")
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
          stn_tst = get_text(text, self.hps)
          with torch.no_grad():
               x_tst = stn_tst.to(device).unsqueeze(0)
               x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)

               speaker_embedding = get_speaker_embedding(ref_audio)
               speaker_embedding = torch.FloatTensor(torch.from_numpy(speaker_embedding)).unsqueeze(0).to(device)

               audio = self.net_g.infer(x_tst, x_tst_lengths, speaker_embedding=speaker_embedding, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

               write(ref_audio.replace(".wav", "_clone.wav"), 22050, audio)

if __name__ == "__main__":
     object = VoiceClone("logs/vivos/G_9000.pth")
     object.infer("hai ba hai ba", "audio/sontung.wav")