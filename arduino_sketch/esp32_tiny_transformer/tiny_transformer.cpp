#include "tiny_transformer.h"
#include <math.h>
#include <string.h>
#include <algorithm>

// masked int8 matmul: A [M x K] * B [K x N]  (B may have block mask)
void matmul_i8_masked(const int8_t* A, float As, const QTensor& B, int M, int K, int N, float* C) {
  for (int i=0;i<M*N;i++) C[i]=0.0f;

  const int br = BLOCK_R, bc = BLOCK_C;
  const int brow = (B.rows + br - 1)/br;
  const int bcol = (B.cols + bc - 1)/bc;

  for (int bi=0; bi<brow; ++bi) {
    for (int bj=0; bj<bcol; ++bj) {
      const int block_id = bi*bcol + bj;
      bool keep = true;
      if (B.mask) {
        uint8_t byte = B.mask[block_id >> 3];
        int bit = block_id & 7;
        keep = ((byte >> bit) & 1);
        if (!keep) continue;
      }

      const int r0 = bi*br;
      const int c0 = bj*bc;
      const int rlen = std::min(br, B.rows - r0);
      const int clen = std::min(bc, B.cols - c0);

      for (int m=0; m<M; ++m) {
        for (int c=0; c<clen; ++c) {
          int acc = 0;
          for (int r=0; r<rlen; ++r) {
            const int a_idx = m*K + (r0 + r);
            const int b_idx = (r0 + r)*N + (c0 + c);
            acc += int(A[a_idx]) * int(B.data[b_idx]);
          }
          C[m*N + (c0 + c)] += (As * B.scale) * float(acc);
        }
      }
    }
  }
}

void matmul_i8_plain(const int8_t* A, float As, const int8_t* B, float Bs, int M, int K, int N, float* C) {
  for (int i=0;i<M*N;i++) C[i]=0.0f;
  for (int m=0; m<M; ++m) {
    for (int n=0;n<N;++n) {
      int acc=0;
      for (int k=0;k<K;++k) acc += int(A[m*K + k]) * int(B[k*N + n]);
      C[m*N + n] = (As * Bs) * float(acc);
    }
  }
}

void linear1_i8(const float* x, const QTensor& W, int IN, int OUT, float* y) {
  static int8_t xq[D_MODEL];
  float xmax=0.f;
  for (int i=0;i<IN;i++) xmax = xmax>fabsf(x[i])?xmax:fabsf(x[i]);
  const float s = xmax/127.0f + 1e-8f;
  for (int i=0;i<IN;i++) xq[i] = (int8_t)roundf(x[i]/s);

  if (W.mask) matmul_i8_masked(xq, s, W, 1, IN, OUT, y);
  else        matmul_i8_plain (xq, s, W.data, W.scale, 1, IN, OUT, y);
}

void layernorm(float* x, const float* g, const float* b, int n) {
  float mean=0.f, var=0.f;
  for (int i=0;i<n;i++) mean += x[i];
  mean /= n;
  for (int i=0;i<n;i++){ float d=x[i]-mean; var += d*d; }
  var /= n;
  const float inv = 1.0f / sqrtf(var + 1e-5f);
  for (int i=0;i<n;i++) x[i] = (x[i]-mean) * inv * g[i] + b[i];
}

void softmax_inplace(float* x, int n) {
  float m = x[0];
  for (int i=1;i<n;i++) if (x[i]>m) m=x[i];
  float s=0.f;
  for (int i=0;i<n;i++){ x[i]=expf(x[i]-m); s+=x[i]; }
  const float inv=1.f/s;
  for (int i=0;i<n;i++) x[i]*=inv;
}

// basic RoPE over pairs
void rope_apply(float* q, float* k, int d, int pos) {
  for (int i=0;i<d;i+=2) {
    float theta = powf(10000.0f, -((float)i/(float)d));
    float a = pos * theta, c = cosf(a), s = sinf(a);
    float q0=q[i], q1=q[i+1], k0=k[i], k1=k[i+1];
    q[i] = q0*c - q1*s; q[i+1] = q0*s + q1*c;
    k[i] = k0*c - k1*s; k[i+1] = k0*s + k1*c;
  }
}

void self_attention_step(const float* x, DecLayer& L, KVCache& cache, int t, float* y) {
  float h[D_MODEL]; memcpy(h, x, sizeof(h));
  layernorm(h, L.ln1_g, L.ln1_b, D_MODEL);

  float q[D_MODEL], kf[D_MODEL], vf[D_MODEL];
  linear1_i8(h, L.Wq, D_MODEL, D_MODEL, q);
  linear1_i8(h, L.Wk_shared, D_MODEL, D_MODEL, kf);
  linear1_i8(h, L.Wv_shared, D_MODEL, D_MODEL, vf);

  rope_apply(q, kf, D_MODEL, t);

  float kmax=0.f, vmax=0.f;
  for (int i=0;i<D_MODEL;i++){ kmax=fmaxf(kmax,fabsf(kf[i])); vmax=fmaxf(vmax,fabsf(vf[i])); }
  if (t==0) { cache.scaleK = kmax/127.0f+1e-8f; cache.scaleV = vmax/127.0f+1e-8f; }
  for (int i=0;i<D_MODEL;i++){
    cache.K[t][i] = (int8_t)roundf(kf[i]/cache.scaleK);
    cache.V[t][i] = (int8_t)roundf(vf[i]/cache.scaleV);
  }
  cache.len = t+1;

  float att_out[D_MODEL]; memset(att_out, 0, sizeof(att_out));
  for (int hId=0; hId<N_HEADS; ++hId) {
    const int off = hId*HEAD_DIM;
    float scores[MAX_SEQ];
    for (int u=0; u<cache.len; ++u) {
      float s=0.f;
      for (int i=0;i<HEAD_DIM;i++)
        s += q[off+i] * (cache.K[u][off+i] * cache.scaleK);
      scores[u] = s / sqrtf((float)HEAD_DIM);
    }
    softmax_inplace(scores, cache.len);
    for (int i=0;i<HEAD_DIM;i++) {
      float sum=0.f;
      for (int u=0;u<cache.len; ++u)
        sum += scores[u] * (cache.V[u][off+i] * cache.scaleV);
      att_out[off+i] = sum;
    }
  }

  float o[D_MODEL]; linear1_i8(att_out, L.Wo, D_MODEL, D_MODEL, o);
  for (int i=0;i<D_MODEL;i++) y[i] = x[i] + o[i];

  float n2[D_MODEL]; memcpy(n2, y, sizeof(n2));
  layernorm(n2, L.ln2_g, L.ln2_b, D_MODEL);
  float h1[D_FF]; linear1_i8(n2, L.W1, D_MODEL, D_FF, h1);
  for (int i=0;i<D_FF;i++) {
    float v=h1[i];
    h1[i] = 0.5f * v * (1.0f + tanhf(0.79788456f*(v + 0.044715f*v*v*v)));
  }
  float h2[D_MODEL]; linear1_i8(h1, L.W2, D_FF, D_MODEL, h2);
  for (int i=0;i<D_MODEL;i++) y[i] += h2[i];
}

void decoder_step(const float* x_emb, DecLayer* layers, int n_layers, KVCache* kvs, int seq_pos, float* logits) {
  float h[D_MODEL]; memcpy(h, x_emb, sizeof(h));
  for (int l=0;l<n_layers;l++) {
    float o[D_MODEL];
    self_attention_step(h, layers[l], kvs[l], seq_pos, o);
    memcpy(h, o, sizeof(h));
  }
  project_to_vocab(h, logits);
}

uint8_t encode_byte(char c) { return (uint8_t)c & 0x7F; }
char    decode_byte(uint8_t t){ return (char)t; }
