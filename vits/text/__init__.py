import sys
sys.path.append("../")
from viphoneme import syms, vi2IPA_split

symbols = syms

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def sequence_to_text(sequence):
    
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            result += _id_to_symbol[symbol_id]     
    return result

def text_to_sequence(text, cleaner_names):

  sequence = []
  text = text.replace('\s+',' ').lower()
  phon = vi2IPA_split(text,"/")
  phon = phon.split("/")[1:]

  eol = -1
  for i,p in reversed(list(enumerate(phon))):
      if p not in ["..",""," ",".","  "]:
          eol = i
          break
  phones = phon[:i+1]+[" ","."]
  phones_id =[]
  for i in phones:
      if i in _symbol_to_id:
          phones_id.append(_symbol_to_id[i])
          #phones_id = [_symbol_to_id[i] for i in phones]
  sequence.extend(phones_id)  

  return sequence



def cleaned_text_to_sequence(cleaned_text):

  sequence = []
  phon = cleaned_text.split("/")[1:]

  eol = -1
  for i,p in reversed(list(enumerate(phon))):
      if p not in ["..",""," ",".","  "]:
          eol = i
          break
  phones = phon[:i+1]+[" ","."]
  phones_id =[]
  for i in phones:
      if i in _symbol_to_id:
          phones_id.append(_symbol_to_id[i])
          #phones_id = [_symbol_to_id[i] for i in phones]
  sequence.extend(phones_id)

  return sequence

if __name__ == "__main__":
  text = "Nơi lưu trữ và cập nhật các bài viết, hình ảnh từ Tuấn Khanh"
  seq = text_to_sequence(text, "")
  print(seq)