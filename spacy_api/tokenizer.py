import re
import sys
import string

if sys.version_info < (3,):
  maketrans = string.maketrans
else:
  maketrans = str.maketrans

def text_to_word_sequence(text,
                          filters='!"\,-()*+/:;<=>?[\\]^`\'|~\t\n',
                          lower=True, split=" ", raw=False):
  """Converts a text to a sequence of word indices.

  # Arguments
      text: Input text (string).
      filters: Sequence of characters to filter out.
      lower: Whether to convert the input to lowercase.
      split: Sentence split marker (string).

  # Returns
      A list of integer word indices.
  """
  if lower:
      text = text.lower()

  # Put spaces
  text = re.sub(r'([!$#@&%])', r' \1 ', text)

  # Add space after .
  text = re.sub(r'([a-zA-Z])(\.)(\s|$)', r'\1 ', text)

  # Remove commas in numbers
  text = re.sub(r'(\d)\,', r'\1', text)
  # Remove dashses
  text = re.sub(r'(OLA.NUMBER|OLA.DATE|\d{1,}|\})(-)', r'\1 ', text, flags=re.IGNORECASE)
  if sys.version_info[0] < 3:
    translate_table = {ord(c): ord(t) for c,t in zip(filters, split*len(filters)) }
  else:
    translate_table = maketrans(filters, split * len(filters))
  text = text.translate(translate_table)
  # Replace extra spaces
  text = re.sub("\s\s+", " ", text)
  if raw:
    return text
  seq = text.split(split)
  return [i for i in seq if i]