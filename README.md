---------------------------------------------------------------------------------------------------
# esp32-tiny-transformer based offline serial chatbot
---------------------------------------------------------------------------------------------------
A minimal **Transformer + BPE tokenizer chatbot** that runs **fully on ESP32-S3** —  
no Wi-Fi, no server, no cloud, even with no internet. Just **on-chip int8 inference** + SD-card weights.

---------------------------------------------------------------------------------------------------
Just type into the Serial-Monitor, the **E32-AI **responds accordingly

---------------------------------------------------------------------------------------------------
## Specification & Features

- **Decoder-only Transformer** with:
  - Multi-head self-attention (RoPE)
  - Quantized int8 matmuls with pruning/masking
  - Optional **ESP-NN acceleration** if available (`#define USE_ESPNN`)
- **BPE tokenizer** (trained offline, runs on MCU)
- **Paged SD storage** implies supports model dimension parameter '128/256` even with limited RAM
- **Streaming chat loop** over Serial Monitor
- Compatible with **ESP32-S3 + PSRAM** boards

---------------------------------------------------------------------------------------------------
## Hardware Requirements

- ESP32-S3 board (recommended with Psudo-SRAM enabled)
- MicroSD card SPI-Module (card capacity ≥ 4GB, FAT32)
- Suitable USB cable

---------------------------------------------------------------------------------------------------
## Arduino IDE setup
- Arduino IDE 1.8.18
- ESP32 boards package 3.1.x
- Libraries:
  - **ArduinoJson** (install via Library Manager)
  - SD (bundled)
- Board: **ESP32-S3-USB-OTG**
-Clock: ESP32-S3 @ 240 MHz
- Flash: "8M with spiffs (3MB APP/1.5MB SPIFFS)"
- USB-OTG: Set to either Enabled or default

---------------------------------------------------------------------------------------------------
## SD contents
Copy the **Python export** output (see `python_toolchain/README_TOOLCHAIN.md`) onto SD-root:
index.json
emb_weight_0000.bin
dec0_Wq_0000.bin
dec0_Wq.mask
dec0_ln1.bin
...
...
...

---------------------------------------------------------------------------------------------------
## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/SaikatMohanta/ESP32-ChatBOT.git


2. Run:
- Upload the INO sketch, open Serial Monitor @ baudrate of **115200**.
- Type a prompt, press Enter.
- The MCU will respond with up to ~80 generated tokens using greedy decoding mechanism.

---------------------------------------------------------------------------------------------------
## Estimating performance & minimum resources (baseline config)
- Config: `L=2, d_model=64, d_ff=256, vocab=128, MAX_SEQ=64`
- Params ≈ ~200k; int8 weights ≈ ~200 KB + (masks/metadata)
- KV cache: 2*(MAX_SEQ*d_model*1B) ≈ 2*64*64 ≈ **8 KB** (int8) + small scales
- Working SRAM during step: ~50–100 KB (buffers + logits + temporary)
- Throughput on ESP32-S3 @ 240 MHz:
  - Reference kernels: a few tokens/s
  - With kernel tuning / ESP-NN: may reach upto **10–70 tokens/s** (model specific and I/O dependent)
-SRAM: ~100 KB free (for K/V cache, temporary buffers, stack)
-PSRAM: recommended but not strictly required for the tiny config
-Flash: Sketch fits well under 3 MB APP partition
-SD card: a few MB is plenty (weights ~200 KB + masks/metadata)

---------------------------------------------------------------------------------------------------
## Potential Tweaks
- Toggle **MQA** (shared K/V) in `tiny_transformer.h` to false for classic MHA/GQA (needs weight export change).
- Change block size `BLOCK_R/C` to match the corresponding pruning exporter.
- To scale up (e.g., d_model=128, L=4, expect ~1–2 MB int8 weights), replace matrix multiplication operations with optimized kernels later (ESP-NN / CMSIS-DSP) for hardware based AI acceleration

---------------------------------------------------------------------------------------------------