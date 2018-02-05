from load import load_data
from util import WordDistribution

vocab2index, verb_phrases, nouns, vocab, cooccur_mtx = load_data("cooccur_mtx")

wd = WordDistribution(cooccur_mtx, vocab2index, vocab)
