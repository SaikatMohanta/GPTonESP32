import os, json, struct, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glove_utils import load_glove_to_vocab, pca_project
import argparse, random

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="../sd_export_sample", help="output dir")
parser.add_argument("--glove", required=False, help="path to glove .txt")
parser.add_argument("--vocab_file", default=None, help="one token per line")
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--vocab", type=int, default=128)
parser.add_argument("--layers", type=int, default=2)
parser.add_argument("--d_ff", type=int, default=256)
parser.add_argument("--page", type=int, default=512)
parser.add_argument("--keep_frac", type=float, default=0.5)
args = parser.parse_args()

OUT = args.out
os.makedirs(OUT, exist_ok=True)
PAGE_SIZE = args.page

D_MODEL = args.d_model
VOCAB   = args.vocab
LAYER_N = args.layers
D_FF    = args.d_ff

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.q = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.o = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.w1 = nn.Linear(D_MODEL, D_FF, bias=False)
        self.w2 = nn.Linear(D_FF, D_MODEL, bias=False)
    def forward(self, x):
        h = self.ln1(x)
        q = self.q(h); k = self.k(h); v = self.v(h)
        x = x + self.o(q)  # toy
        h = self.ln2(x)
        x = x + self.w2(torch.nn.functional.gelu(self.w1(h)))
        return x

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYER_N))
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.emb.weight
    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks: x = b(x)
        return self.head(x)

def quantize(t):
    a = t.astype(np.float32)
    s = float(np.max(np.abs(a))) / 127.0 + 1e-8
    q = np.round(a / s).astype(np.int8)
    return q, np.float32(s)

def write_paged(name, raw):
    pages=[]
    for i in range(0, len(raw), PAGE_SIZE):
        chunk = raw[i:i+PAGE_SIZE]
        if len(chunk) < PAGE_SIZE:
            chunk += bytes(PAGE_SIZE - len(chunk))
        fname = f"{name}_{i//PAGE_SIZE:04d}.bin"
        with open(os.path.join(OUT, fname), "wb") as f: f.write(chunk)
        pages.append(fname)
    return pages

def mask_4x4(arr, keep_frac=0.5):
    br=4; bc=4
    r,c = arr.shape
    brow=(r+br-1)//br
    bcol=(c+bc-1)//bc
    scores = np.zeros((brow,bcol), dtype=np.float32)
    for bi in range(brow):
        for bj in range(bcol):
            r0=bi*br; c0=bj*bc
            blk = arr[r0:r0+br, c0:c0+bc]
            scores[bi,bj] = np.sum(np.abs(blk))
    flat = scores.flatten()
    k = int(max(1, keep_frac*flat.size))
    thr = np.sort(flat)[-k]
    keep = (scores >= thr).astype(np.uint8).flatten()
    out = bytearray((keep.size+7)//8)
    for i,b in enumerate(keep):
        if b: out[i>>3] |= (1 << (i&7))
    return bytes(out)

def main():
    model = TinyModel()
    # Optionally init embeddings from GloVe -> project to d_model
    if args.glove:
        vocab = [chr(i) for i in range(VOCAB)]
        glove = load_glove_to_vocab(args.glove, vocab, min(100, D_MODEL))
        if glove.shape[1] != D_MODEL:
            P = pca_project(glove, D_MODEL)
            glove = glove @ P
        with torch.no_grad():
            model.emb.weight.copy_(torch.tensor(glove))

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    data = torch.randint(0, VOCAB, (64, 64))
    for it in range(200):
        logits = model(data)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), data.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0: print("it", it, "loss", float(loss))

    index = {}

    # Embedding
    E = model.emb.weight.detach().cpu().numpy()
    qE, sE = quantize(E)
    raw = qE.tobytes() + struct.pack("<f", sE)
    index["emb_weight"] = write_paged("emb_weight", raw)
    index["emb_weight_shape"] = [E.shape[0], E.shape[1]]

    # Layers
    for li in range(LAYER_N):
        blk = model.blocks[li]
        tensors = {
            f"dec{li}_Wq": blk.q.weight.detach().cpu().numpy(),
            f"dec{li}_Wk_shared": blk.k.weight.detach().cpu().numpy(),
            f"dec{li}_Wv_shared": blk.v.weight.detach().cpu().numpy(),
            f"dec{li}_Wo": blk.o.weight.detach().cpu().numpy(),
            f"dec{li}_W1": blk.w1.weight.detach().cpu().numpy(),
            f"dec{li}_W2": blk.w2.weight.detach().cpu().numpy(),
        }
        for name, arr in tensors.items():
            q, s = quantize(arr)
            raw = q.tobytes() + struct.pack("<f", s)
            index[name] = write_paged(name, raw)
            index[name+"_shape"] = [arr.shape[0], arr.shape[1]]
            # mask
            m = mask_4x4(arr, keep_frac=args.keep_frac)
            mname = name + ".mask"
            with open(os.path.join(OUT, mname), "wb") as f: f.write(m)
            index[mname] = [mname]
        # layernorm params
        ln1 = np.concatenate([blk.ln1.weight.detach().cpu().numpy().astype(np.float32),
                              blk.ln1.bias.detach().cpu().numpy().astype(np.float32)])
        ln2 = np.concatenate([blk.ln2.weight.detach().cpu().numpy().astype(np.float32),
                              blk.ln2.bias.detach().cpu().numpy().astype(np.float32)])
        ln1f = f"dec{li}_ln1.bin"; ln2f = f"dec{li}_ln2.bin"
        with open(os.path.join(OUT, ln1f),"wb") as f: f.write(ln1.tobytes())
        with open(os.path.join(OUT, ln2f),"wb") as f: f.write(ln2.tobytes())
        index[ln1f] = [ln1f]; index[ln2f] = [ln2f]

    with open(os.path.join(OUT, "index.json"), "w") as jf:
        json.dump(index, jf, indent=2)

    print("Exported pages ->", OUT)

if __name__ == "__main__":
    main()
