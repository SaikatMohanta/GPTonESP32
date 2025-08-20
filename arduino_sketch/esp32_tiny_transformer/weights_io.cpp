#include "weights_io.h"
#include <ArduinoJson.h>   // Install via Library Manager (ArduinoJson by Benoit Blanchon)
#include <string.h>

StaticJsonDocument<8192> g_index;
static QTensor g_emb = {nullptr, 1.0f, nullptr, VOCAB, D_MODEL};
static bool g_emb_loaded = false;

static bool load_qtensor_from_index(const char* key, QTensor& out) {
  if (!g_index.containsKey(key)) return false;
  JsonArray pages = g_index[key].as<JsonArray>();
  if (pages.isNull() || pages.size()==0) return false;

  // We assume small tensors fit in a single page (as per exporter). Scale stored at page end.
  const char* fname = pages[0];
  const uint8_t* page = SDV.readPageToBuffer(fname);
  if (!page) return false;

  const size_t pageSz = SDV.getPageSize();
  float scale;
  memcpy(&scale, page + pageSz - 4, 4);

  out.data  = (const int8_t*)page;
  out.scale = scale;
  out.mask  = nullptr;

  // shape
  String shapeKey = String(key) + "_shape";
  if (g_index.containsKey(shapeKey)) {
    JsonArray sh = g_index[shapeKey].as<JsonArray>();
    out.rows = sh[0]; out.cols = sh[1];
  } else {
    out.rows = D_MODEL; out.cols = D_MODEL;
  }

  // mask (optional)
  String maskKey = String(key) + ".mask";
  if (g_index.containsKey(maskKey)) {
    JsonArray mpages = g_index[maskKey].as<JsonArray>();
    const char* mfile = mpages[0];
    out.mask = SDV.readPageToBuffer(mfile);
  }

  return true;
}

bool load_metadata_and_embeddings() {
  size_t sz = SDV.readFileToBuffer("index.json", nullptr);
  if (sz == 0) return false;

  char* tmp = (char*)malloc(sz+1);
  if (!tmp) return false;
  SDV.readFileToBuffer("index.json", tmp);
  tmp[sz]=0;

  auto err = deserializeJson(g_index, tmp);
  free(tmp);
  if (err) return false;

  // embeddings
  if (!load_qtensor_from_index("emb_weight", g_emb)) return false;
  g_emb_loaded = true;
  return true;
}

bool load_decoder_layer(int layer_id, DecLayer& out) {
  char key[48];

  snprintf(key, sizeof(key), "dec%d_Wq", layer_id);
  if (!load_qtensor_from_index(key, out.Wq)) return false;

  snprintf(key, sizeof(key), "dec%d_Wk_shared", layer_id);
  if (!load_qtensor_from_index(key, out.Wk_shared)) return false;

  snprintf(key, sizeof(key), "dec%d_Wv_shared", layer_id);
  if (!load_qtensor_from_index(key, out.Wv_shared)) return false;

  snprintf(key, sizeof(key), "dec%d_Wo", layer_id);
  if (!load_qtensor_from_index(key, out.Wo)) return false;

  snprintf(key, sizeof(key), "dec%d_W1", layer_id);
  if (!load_qtensor_from_index(key, out.W1)) return false;

  snprintf(key, sizeof(key), "dec%d_W2", layer_id);
  if (!load_qtensor_from_index(key, out.W2)) return false;

  // layernorm params (float32 arrays concatenated)
  String ln1 = String("dec") + String(layer_id) + "_ln1.bin";
  const uint8_t* p1 = SDV.readPageToBuffer(ln1.c_str());
  if (!p1) return false;
  memcpy(out.ln1_g, p1, sizeof(float)*D_MODEL);
  memcpy(out.ln1_b, p1 + sizeof(float)*D_MODEL, sizeof(float)*D_MODEL);

  String ln2 = String("dec") + String(layer_id) + "_ln2.bin";
  const uint8_t* p2 = SDV.readPageToBuffer(ln2.c_str());
  if (!p2) return false;
  memcpy(out.ln2_g, p2, sizeof(float)*D_MODEL);
  memcpy(out.ln2_b, p2 + sizeof(float)*D_MODEL, sizeof(float)*D_MODEL);

  return true;
}

void embedding_lookup(uint8_t token, float* out_emb) {
  if (!g_emb_loaded) return;
  const int8_t* row = g_emb.data + (size_t)token * g_emb.cols;
  for (int i=0;i<D_MODEL;i++) out_emb[i] = float(row[i]) * g_emb.scale;
}

void project_to_vocab(const float* x, float* logits) {
  // tied weights: logits = E * x  (E is [V x D])
  for (int v=0; v<VOCAB; ++v) {
    const int8_t* row = g_emb.data + v*g_emb.cols;
    float s=0.f;
    for (int i=0;i<D_MODEL;i++) s += float(row[i]) * g_emb.scale * x[i];
    logits[v] = s;
  }
}
