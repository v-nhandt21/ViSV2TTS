from re import A
from transformers.file_utils import cached_path, hf_bucket_url
import os, zipfile
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import torch
import kenlm
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
import os
from multiprocessing import Pool
import argparse, subprocess, tempfile

def extract_audio(filename, channels=1, rate=16000):
     """
     Extract audio from an input file to a temporary WAV file.
     """
     temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
     if not os.path.isfile(filename):
          print("The given file does not exist: {}".format(filename))
          raise Exception("Invalid filepath: {}".format(filename))

     command = ["ffmpeg", "-y", "-i", filename,
                    "-ac", str(channels), "-ar", str(rate),
                    "-loglevel", "error", temp.name]
     use_shell = True if os.name == "nt" else False
     subprocess.check_output(command, stdin=open(os.devnull), shell=use_shell)
     return temp.name, rate

class Wav2Vec:
     def __init__(self):
          
          self.device = "cuda"
          # Load Wav2Vec
          cache_dir = './cache/'
          self.processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h", cache_dir=cache_dir)
          lm_file = hf_bucket_url("nguyenvulebinh/wav2vec2-base-vietnamese-250h", filename='vi_lm_4grams.bin.zip')
          lm_file = cached_path(lm_file,cache_dir=cache_dir)
          with zipfile.ZipFile(lm_file, 'r') as zip_ref:
               zip_ref.extractall(cache_dir)
          lm_file = cache_dir + 'vi_lm_4grams.bin'
          self.model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h", cache_dir=cache_dir)
          self.model.to(self.device)

          # Load Ngram LM
          self.ngram_lm_model = self.get_decoder_ngram_model(self.processor.tokenizer, lm_file)

     def get_decoder_ngram_model(self, tokenizer, ngram_lm_path):
          vocab_dict = tokenizer.get_vocab()
          sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
          vocab = [x[1] for x in sort_vocab][:-2]
          vocab_list = vocab
          # convert ctc blank character representation
          vocab_list[tokenizer.pad_token_id] = ""
          # replace special characters
          vocab_list[tokenizer.unk_token_id] = ""
          # vocab_list[tokenizer.bos_token_id] = ""
          # vocab_list[tokenizer.eos_token_id] = ""
          # convert space character representation
          vocab_list[tokenizer.word_delimiter_token_id] = " "
          # specify ctc blank char index, since conventially it is the last entry of the logit matrix
          alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=tokenizer.pad_token_id)
          lm_model = kenlm.Model(ngram_lm_path)
          decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(lm_model))
          return decoder

     # define function to read in sound file
     def map_to_array(self, batch):
          speech, sampling_rate = sf.read(batch["file"])
          batch["speech"] = speech
          batch["sampling_rate"] = sampling_rate
          return batch

     def inference(self, filename):

          # load dummy dataset and read soundfiles
          ds = self.map_to_array({"file": filename})

          # infer model
          input_values = self.processor(ds["speech"], sampling_rate=ds["sampling_rate"], return_tensors="pt").input_values
          input_values = input_values.to(self.device)
          # model.to("cuda")
          logits = self.model(input_values).logits[0]
          # print(logits.shape)

          # decode ctc output
          pred_ids = torch.argmax(logits, dim=-1)
          greedy_search_output = self.processor.decode(pred_ids)
          beam_search_output = self.ngram_lm_model.decode(logits.cpu().detach().numpy(), beam_width=500)
          # print("Greedy search output: {}".format(greedy_search_output))
          # print("Beam search output: {}".format(beam_search_output))
          return beam_search_output

if __name__ == "__main__":
     w2v = Wav2Vec()
     import glob, tqdm

     parser = argparse.ArgumentParser()
     parser.add_argument('--wavs', default="DATA/wavs", help="", type=str)
     parser.add_argument('--train_file', default="DATA/train.txt", help="", type=str)
     parser.add_argument('--val_file', default="DATA/train.txt", help="", type=str)
     args = parser.parse_args()

     os.makedirs(os.path.dirname(args.train_file), exist_ok = True)

     count_val = 0

     fw = open(args.train_file, "w+", encoding="utf-8")
     fw_val = open(args.val_file, "w+", encoding="utf-8")
     for i in tqdm.tqdm(glob.glob(args.wavs + "/*.wav")):
          audio_filename, audio_rate = extract_audio(i)
          output = w2v.inference(audio_filename)
          fw.write(i.split("/")[-1] + " " + output + "\n")

          if count_val < 64:
               count_val = count_val + 1
               fw_val.write(i.split("/")[-1] + " " + output + "\n")

     fw.close()
     fw_val.close()