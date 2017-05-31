import cachetools.func
import spacy

nlp_objects = {}

DEFAULT_ATTRIBUTES = ("text", "lemma_", "pos_", "tag_", "vector")
SERVER_CACHE_SIZE = 100


def get_nlp(model="en", embeddings_path=None):
    if embeddings_path not in nlp_objects:
        if embeddings_path is None:
            nlp_ = spacy.load(model)
        else:
            nlp_ = spacy.load(model, vectors=embeddings_path)
        nlp_objects[embeddings_path] = nlp_
    return nlp_objects[embeddings_path]


def json_safety(token, x):
    try:
        value = getattr(token, x)
    except AttributeError:
        print(x, "not found on spacy object")
        value = "ERROR"
    if isinstance(value, (str, int, bool)):
        return value
    else:
        # vectors
        return [float(e) for e in value]


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
    nlp_ = get_nlp(model, embeddings_path)
    if local:
        sentences = nlp_(document)
    else:
        sentences = []
        for sent in nlp_(document).sents:
            sentence = [{x: json_safety(token, x) for x in attributes}
                        for token in sent]
            sentences.append(sentence)
    return sentences


def bulk(documents, model="en", embeddings_path=None, attributes=None, local=False):
    attributes = convert_attr(attributes)
    parsed_documents = [single(d, model, embeddings_path, attributes, local)
                        for d in documents]
    return parsed_documents
