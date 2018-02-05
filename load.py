import dill as pickle
from scipy.io import loadmat

def load_data(filename):
  with open(filename+".pkl", "rb") as f:
    data = pickle.load(f)
  cooccur_mtx = loadmat(filename+".pkl.mat")
  return data["vocab2index"], data["verb_phrases"], data["nouns"], data["vocab"], cooccur_mtx['cooccur']
