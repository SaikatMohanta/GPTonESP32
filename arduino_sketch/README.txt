# ESP32-S3 Tiny Transformer 

This sketch runs a tiny Transformer decoder on **ESP32-S3** and chats over the Serial Monitor.
Weights are **int8**, stored on a microSD as 512-byte "pages"; masks implement **block pruning**.
K/V cache is kept in SRAM; **MQA** (shared K,V) reduces memory bandwidth.

## Hardware
- ESP32-S3 dev board (with PSRAM recommended)
- MicroSD wired to SPI (or SDMMC if your board supports it)
- Format SD: FAT32
- Capacity: 4GB or more

## Run
- Upload the sketch, open Serial Monitor @ **115200**.
- Type a prompt, press Enter.
- The MCU will respond with up to ~80 generated tokens using greedy decoding
