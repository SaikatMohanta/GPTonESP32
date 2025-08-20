import numpy as np

def load_glove_to_vocab(glove_path, vocab_list, expected_dim):
    vecs = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            if len(parts) < 10:  # skip bad lines
                continue
            w = parts[0]
            vals = np.array(parts[1:], dtype=np.float32)
            vecs[w] = vals
    V = len(vocab_list)
    d = expected_dim
    out = np.random.normal(scale=0.01, size=(V, d)).astype(np.float32)
    for i,w in enumerate(vocab_list):
        if w in vecs and vecs[w].shape[0] == d:
            out[i,:] = vecs[w]
    return out

def pca_project(W, d_out):
    Wc = W - W.mean(0)
    U,S,Vt = np.linalg.svd(Wc, full_matrices=False)
    P = Vt[:d_out].T.astype(np.float32)
    return P
