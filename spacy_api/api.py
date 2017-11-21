import cachetools.func
import spacy
from spacy_api.tokenizer import text_to_word_sequence
from spacy_api.negation import is_neg

nlp_objects = {}

DEFAULT_ATTRIBUTES = ("text", "lemma_", "pos_", "tag_", "vector")
DEFAULT_ENTITY_ATTRIBUTES = ('start_char', 'label_', 'string', 'text', 'end_char')
SERVER_CACHE_SIZE = 100

# Custom tokenizer to Spacy
class Tokenizer(object):
    def __init__(self, nlp):
        self.vocab = nlp.vocab

    def __call__(self, text):
        # words = text_to_word_sequence(text)
        words = text.split()
        spaces = [True] * len(words)
        return spacy.tokens.Doc(self.vocab, words=words, spaces=spaces)


def get_nlp(model="en", embeddings_path=None):
    if embeddings_path not in nlp_objects:
        if embeddings_path is None:
            nlp_ = spacy.load(model)
        else:
            nlp_ = spacy.load(model, vectors=embeddings_path)
        nlp_objects[embeddings_path] = nlp_
    return nlp_objects[embeddings_path]

def isgenerator(iterable):
    return hasattr(iterable,'__iter__') and not hasattr(iterable,'__len__')

def recursive_iterate(enum):
  value = []
  for v in enum:
    if (isgenerator(v)):
      value.append(recursive_iterate(v))
    else:
      value.append(str(v))
  return value

def json_safety(token, x):
    try:
        value = getattr(token, x)
    except AttributeError:
        print(x, "not found on spacy object")
        value = "ERROR"
    if isinstance(value, (str, float, int, bool)):
        return value
    else:
        if x == 'head':
          value = str(value)
        if x == 'vector':
          # vectors
          return [float(e) for e in value]
        if (isgenerator(value)):
          value = recursive_iterate(value)
        return value


def convert_attr(attributes):
    if attributes is None:
        attributes = DEFAULT_ATTRIBUTES
    elif isinstance(attributes, str):
        attributes = tuple(attributes.split(","))
    return attributes


def vectors_length(model="en", embeddings_path=None):
    nlp_ = get_nlp(model, embeddings_path)
    return nlp_.vocab.vectors_length


def description(model="en", embeddings_path=None):
    nlp_ = get_nlp(model, embeddings_path)
    return "{}_{}_{}".format(nlp_.meta['lang'], nlp_.meta['name'], nlp_.meta['version'])


@cachetools.func.lru_cache(maxsize=SERVER_CACHE_SIZE)
def single(document, model="en", embeddings_path=None, attributes=None, local=False):
    attributes = convert_attr(attributes)
    print('called top', model, embeddings_path)
    nlp_ = get_nlp(model, embeddings_path)
    print('called bottom')
    if local:
        sentences = nlp_(document)
    else:
        sentences = []
        for sent in nlp_(document).sents:
            sentence = [{x: json_safety(token, x) for x in attributes}
                        for token in sent]
            sentences.append(sentence)
    return sentences

@cachetools.func.lru_cache(maxsize=SERVER_CACHE_SIZE)
def entity(document, model="en", embeddings_path=None, local=False):
    attributes = DEFAULT_ENTITY_ATTRIBUTES
    nlp_ = get_nlp(model, embeddings_path)
    ents = []
    if local:
        ents = nlp_(document).ents
    else:
        ents = [{x: json_safety(token, x) for x in attributes} for token in nlp_(document).ents]
    return ents

def negation(document, slotIndex, model="en", embeddings_path=None, local=False):
    nlp_ = get_nlp(model, embeddings_path)
    sentences = nlp_(document)
    return is_neg(sentences, slotIndex)

def bulk(documents, model="en", embeddings_path=None, attributes=None, local=False):
    attributes = convert_attr(attributes)
    parsed_documents = [single(d, model, embeddings_path, attributes, local)
                        for d in documents]
    return parsed_documents
