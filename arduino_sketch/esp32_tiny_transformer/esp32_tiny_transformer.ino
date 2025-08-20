// esp32_tiny_transformer.ino
// ESP32-S3 + microSD-paged tiny Transformer demo usage example code(serial chatbot)

#include <Arduino.h>
#include "sd_vram_adapter.h"
#include "weights_io.h"
#include "tiny_transformer.h"

#define N_DEC_LAYERS 2
#define MAX_GEN_TOKENS 80

DecLayer g_layers[N_DEC_LAYERS];
KVCache  g_kvc[N_DEC_LAYERS];

void reset_kv() { for (int i=0;i<N_DEC_LAYERS;i++) g_kvc[i].len = 0; }

void setup() {
  Serial.begin(115200);
  while(!Serial) { delay(10); }
  Serial.println("\n[ESP32-S3] Tiny Transformer booting...");

  if (!SDV.begin()) {
    Serial.println("ERROR: SDV.begin() failed (check SD card wiring/format).");
    while(true) delay(1000);
  }
  Serial.printf("SDV OK, PAGE_SIZE=%u bytes\n", (unsigned)SDV.getPageSize());

  if (!load_metadata_and_embeddings()) {
    Serial.println("ERROR: failed to load index.json / embeddings.");
    while(true) delay(1000);
  }

  for (int i=0;i<N_DEC_LAYERS;i++) {
    if (!load_decoder_layer(i, g_layers[i])) {
      Serial.printf("ERROR: failed to load decoder layer %d\n", i);
      while(true) delay(1000);
    }
    g_kvc[i].len = 0;
  }

  Serial.println("Model ready. Type a line and press Enter.");
}

String line;

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      if (line.length() == 0) { Serial.print("> "); continue; }

      reset_kv();

      float emb[D_MODEL];
      float logits[VOCAB];

      // feed prompt
      int t = 0;
      for (size_t i=0; i<line.length() && t<MAX_SEQ; ++i, ++t) {
        uint8_t tok = encode_byte(line[i]);
        embedding_lookup(tok, emb);
        decoder_step(emb, g_layers, N_DEC_LAYERS, g_kvc, t, logits);
      }

      // generate reply via greedy mechanism 
      //(greedy algorithms are relatively easy to implement, though it does not always guarantee the best possible solution in general
      
      Serial.print("E32-AI:  ");
      for (int i=0; i<MAX_GEN_TOKENS && t<MAX_SEQ; ++i, ++t) {
        int argmax = 0; float best = logits[0];
        for (int v=1; v<VOCAB; ++v) if (logits[v] > best) { best = logits[v]; argmax = v; }
        char outc = decode_byte((uint8_t)argmax);
        Serial.print(outc);
        embedding_lookup((uint8_t)argmax, emb);
        decoder_step(emb, g_layers, N_DEC_LAYERS, g_kvc, t, logits);
      }
      Serial.println();
      line = "";
      Serial.print(">>>  ");
    } else {
      line += c;
    }
  }
  delay(2);
}
