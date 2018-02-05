import numpy

class WordDistribution:

  def __init__(self, cooccur_mtx, vocab2index, vocab_counts):
    self.cooccur_mtx = cooccur_mtx
    self.vocab2index = vocab2index
    self.index2vocab = {v:k for k,v in vocab2index.items()}
    self.vocab_counts = vocab_counts
    self.vocab_size = len(vocab_counts)
    total_count = float(sum([v for k,v in vocab_counts.items()]))
    self.vocab_dist = numpy.array([vocab_counts[self.index2vocab[i]] for i in range(0,self.vocab_size)]) / total_count

  def sample(self, condition=None, n=None, as_string=False):
    if not condition:
      choice = numpy.random.choice(range(0,self.vocab_size), size=n, p=self.vocab_dist)
    else:
      if isinstance(condition, str):
        condition = self.vocab2index[condition]
      dist = self.cooccur_mtx[condition,:].toarray().flatten()
      p_dist = dist / dist.sum()
      choice = numpy.random.choice(range(0,self.vocab_size), size=n, p=p_dist)
    if n:
      if as_string:
        return [self.index2vocab[x] for x in choice]
    else:
      if as_string:
        return self.index2vocab[choice]
    return choice



