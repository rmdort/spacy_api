from nltk import Tree
import numpy as np

neg_words = ['don\'t', 'dont', 'haven\'t', 'havent', 'hasn\'t', 'not', 'no', 'never', 'nope', 'nor']
semi_neg_words = ['but', 'although', 'though', 'yet']

def _parse_dependency_tree(doc):
  # try to get to root without encountering neg
  root = [w for w in doc if w.head == w][0]
  t2i = {doc[i]: i for i in range(len(doc))}
  negs = []
  for w in doc:
    sn = False
    orig_w = w
    while True:
      if w.text in neg_words:
        negs.append(t2i[orig_w])
      if 'V' in w.tag_ or True:
        neg = False
        for c in w.children:
          if c.text in neg_words:
            if not sn:
              negs.append(t2i[orig_w])
              neg = True
            break
          elif t2i[c] < t2i[orig_w]:
            if c.text in semi_neg_words:
              sn = True
        if neg:
          break
      if w == w.head:
        break
      else:
        w = w.head
  return negs

def is_neg(nlp_doc, index):
  negs = _parse_dependency_tree(nlp_doc)
  if type(index) in (tuple, list):
    return [idx in negs for idx in index]
  else:
    return index in negs


def to_nltk_tree(node):
  if node.n_lefts + node.n_rights > 0:
    return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
  else:
    return node.orth_


def disp_tree(text):
  [to_nltk_tree(sent.root).pretty_print() for sent in nlp(text).sents]
