#pragma once
#include <Arduino.h>

// ====== Model dimensions===========================================================================
constexpr int D_MODEL = 64;
constexpr int N_HEADS = 2;
constexpr int HEAD_DIM = D_MODEL / N_HEADS;
constexpr int D_FF   = 256;
constexpr int MAX_SEQ= 64;
constexpr int VOCAB  = 128;

// ======= Features ================================================================================
constexpr bool USE_MQA = true;   // true implies Multi-Query Attention (shared Key/Value)
constexpr int  BLOCK_R = 4;      // block pruning tile
constexpr int  BLOCK_C = 4;

// ======= Quant tensor ================================================================================
struct QTensor {
  const int8_t*  data;   // row-major  KxN
  float          scale;  // dequant scale for data
  const uint8_t* mask;   // bit-packed mask for BLOCK_R x BLOCK_C blocks (though optional)
  int rows, cols;
};

struct DecLayer {
  QTensor Wq;
  QTensor Wk_shared;   // used if USE_MQA == true
  QTensor Wv_shared;
  QTensor Wo;
  QTensor W1;          // Feed Forward Network
  QTensor W2;
  float ln1_g[D_MODEL], ln1_b[D_MODEL];
  float ln2_g[D_MODEL], ln2_b[D_MODEL];
};

struct KVCache {
  // per-time-step cached Key/Value (int8) for attention
  int8_t K[MAX_SEQ][D_MODEL];
  int8_t V[MAX_SEQ][D_MODEL];
  float  scaleK, scaleV;
  int    len;
  KVCache(): scaleK(1e-6f), scaleV(1e-6f), len(0) {}
};

// ================== Kernels ========================================================================================================
void matmul_i8_masked(const int8_t* A, float As, const QTensor& B, int M, int K, int N, float* C);
void matmul_i8_plain(const int8_t* A, float As, const int8_t* B, float Bs, int M, int K, int N, float* C);
void linear1_i8(const float* x, const QTensor& W, int IN, int OUT, float* y);
void layernorm(float* x, const float* g, const float* b, int n);
void softmax_inplace(float* x, int n);
void rope_apply(float* q, float* k, int d, int pos);

void self_attention_step(const float* x, DecLayer& L, KVCache& cache, int seq_pos, float* y);
void mlp_block(const float* x, DecLayer& L, float* y);
void decoder_step(const float* x_emb, DecLayer* layers, int n_layers, KVCache* kvs, int seq_pos, float* logits);

// ==============byte tokenizer=====================================================================================================
uint8_t encode_byte(char c);
char    decode_byte(uint8_t t);

// vocab head (declared in weights_io)
void project_to_vocab(const float* x, float* logits);
