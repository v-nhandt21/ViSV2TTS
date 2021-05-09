import os
from os.path import exists, join, basename, splitext
import sys
import numpy as np

from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import soundfile as sf


encoder.load_model(Path("encoder/saved_models/pretrained.pt"))
synthesizer = Synthesizer(Path("synthesizer/saved_models/pretrained/pretrained.pt"))
vocoder.load_model(Path("vocoder/saved_models/pretrained/pretrained.pt"))

def getEmbedding(audio_path):
  embedding = encoder.embed_utterance(encoder.preprocess_wav(audio_path, "16000"))
  return embedding

def synthesize(embed, text):
  specs = synthesizer.synthesize_spectrograms([text], [embed])
  generated_wav = vocoder.infer_waveform(specs[0])
  generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

  print(synthesize.sample_rate)
  sf.write('infer5.wav', generated_wav, synthesizer.sample_rate, 'PCM_24')

text = "One of the two people who tested positive for the novel coronavirus in the United Kingdom is a student at the University of York in northern England." #@param {type:"string"}

embedding = getEmbedding("5.wav")
synthesize(embedding, text)