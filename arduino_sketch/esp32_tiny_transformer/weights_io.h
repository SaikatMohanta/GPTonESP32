#pragma once
#include <Arduino.h>
#include "tiny_transformer.h"
#include "sd_vram_adapter.h"

// index & loading
bool load_metadata_and_embeddings();
bool load_decoder_layer(int layer_id, DecLayer& out);

// embedding + head
void embedding_lookup(uint8_t token, float* out_emb);
void project_to_vocab(const float* x, float* logits);
