# ESP32-S3 Tiny Transformer — SD-paged, pruned+quantized

This sketch runs a tiny Transformer decoder on **ESP32-S3** and chats over the Serial Monitor.
Weights are **int8**, stored on a microSD as 512-byte "pages"; masks implement **block pruning**.
K/V cache is kept in SRAM; **MQA** (shared K,V) reduces memory bandwidth.

## Hardware
- ESP32-S3 dev board (with PSRAM recommended)
- microSD wired to SPI (or SDMMC if your board supports it)
- Format SD: FAT32

## Arduino setup
- Arduino IDE 1.8.18
- ESP32 boards package 3.1.0
- Libraries:
  - **ArduinoJson** (install via Library Manager)
  - SD (bundled)
- Board: **ESP32-S3-USB-OTG**
- Flash: "8M with spiffs (3MB APP/1.5MB SPIFFS)"
- USB-OTG: Enabled or default
- PSRAM: Enabled if available

## SD contents
Copy the **Python export** output (see `python_toolchain/README_TOOLCHAIN.md`) onto SD root:
index.json
emb_weight_0000.bin
dec0_Wq_0000.bin
dec0_Wq.mask
dec0_ln1.bin
..............................
..............................


## Run
- Upload the sketch, open Serial Monitor @ **115200**.
- Type a prompt, press Enter.
- The MCU will respond with up to ~80 generated tokens using greedy decoding.

## Build/compile fixes integrated
- std::min typed: `std::min<size_t>()` to avoid unsigned type mismatch.
- Added `syncMetadata` stub to `SDVirtualRAM`.
- Fixed stray `else` in debugger and centralized `pageMeta` in `SDVirtualRAM_Meta.cpp`.
- Ensured headers include order so `DecLayer` is known to `weights_io`.

## Performance & resource estimates (baseline)
- Config: `L=2, d_model=64, d_ff=256, vocab=128, MAX_SEQ=64`
- Params ≈ ~200k → int8 weights ≈ ~200 KB
- KV cache: 2*(MAX_SEQ*d_model*1B) ≈ 2*64*64 ≈ **8 KB** (int8) + small scales
- Working SRAM during step: ~50–100 KB (buffers + logits + temp)
- Throughput on ESP32-S3 @ 240 MHz:
  - Reference kernels: few tokens/s
  - With kernel tuning / ESP-NN: **10–80 tokens/s** (model and IO dependent)

## Tweaks
- Toggle **MQA** (shared K/V) in `tiny_transformer.h` to false for classic MHA/GQA (needs weight export change).
- Change block size `BLOCK_R/C` to match your pruning exporter.
- Replace matmul with optimized kernels later (ESP-NN / CMSIS-DSP) for speed.


************************************************************************************************************************************************************************************************************
These scripts produce the SD files (int8 pages + masks + index.json).
************************************************************************************************************************************************************************************************************
