import copy
import math
import pickle
import numpy as np
from collections import defaultdict
import os
from six.moves import xrange
import six

def precook(s, n=4, out=False):
    words = s.split()
    counts = defaultdict(int)
    for k in xrange(1, n+1):
        for i in xrange(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): 
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    return precook(test, n, True)

class Cider(object):
    """CIDEr scorer."""

    def copy(self):
        new = Cider(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0, idf=None):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.ref_len = None
        
        if idf:
            self.document_frequency = idf['df']
            self.ref_len = np.log(float(idf['ref_len']))  # Use reference length from the IDF
        
        self.cook_append(test, refs)

    def cook_append(self, test, refs):
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test))  
            else:
                self.ctest.append(None)  

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        if type(other) is tuple:
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self):
        '''Compute term frequency for reference data to generate IDF.'''
        if not self.document_frequency:  # Handle empty DF (for 'corpus' mode)
            for refs in self.crefs:
                for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                    self.document_frequency[ngram] += 1

    def compute_cider(self, df_mode):
        def counts2vec(cnts):
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                df = np.log(max(1.0, self.document_frequency[ngram]))
                n = len(ngram) - 1
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            delta = float(length_hyp - length_ref)
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                for (ngram, count) in vec_hyp[n].items():
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n] * norm_ref[n])

                val[n] *= np.e**(-(delta**2) / (2 * self.sigma**2)) 
            return val

        if df_mode == "corpus":
            self.ref_len = np.log(float(len(self.crefs)))  # Use total references in corpus as ref_len

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score)
            score_avg /= len(refs)  
            score_avg *= 10.0 
            scores.append(score_avg)
        return scores

    def compute_score(self, df_mode, option=None, verbose=0):
        '''Compute the CIDEr score based on df_mode (corpus or IDF-based).'''
        self.compute_doc_freq()

        if df_mode == "corpus":
            if not self.document_frequency:  # Handle the case where DF is empty
                raise ValueError("Document frequency is empty. Please check the corpus data.")

        min_required_data = max(self.document_frequency.values())
        # print(min_required_data)# For corpus mode, we require at least one reference
        # if len(self.ctest) < min_required_data:
        #     raise ValueError(f"Insufficient test data: {len(self.ctest)} samples, but at least {min_required_data} are required.")
        
        score = self.compute_cider(df_mode)
        return np.mean(np.array(score)), np.array(score)
