************************************************************************************************

# Python exporter — tiny transformer pages

Create the SD card contents for the ESP32 sketch.

## Setup:	

cd python_toolchain
python -m venv venv
source venv/bin/activate # on Windows: venv\Scripts\activate
pip install -r requirements.txt

## Run (with optional GloVe init)
python export_train_quant.py --glove D:\path\to\glove.6B.100d.txt --out ../sd_export_sample

This writes `index.json`, `*.bin` pages, and `*.mask` to `../sd_export_sample`.

Copy all files in `sd_export_sample` to the **root of your SD card**.

### Notes
- Page size defaults to 512 to align with SD sectors.
- Structured block pruning masks (4×4) are exported as `.mask`.
- The Arduino code assumes each weight fits in a **single** page; this is true for the small config. If you scale up, keep per-tensor under 512 bytes or extend the loader to handle multiple pages.


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Estimating minimum hardware/resources (baseline config)

SRAM: ~100 KB free (for K/V cache, temporary buffers, stack).

Flash: Sketch fits well under 3 MB APP partition.

SD card: a few MB is plenty (weights ~200 KB + masks/metadata).

Clock: ESP32-S3 @ 240 MHz.

PSRAM: recommended but not strictly required for the tiny config.

To scale up (e.g., d_model=128, L=4), expect ~1–2 MB int8 weights; you’ll want PSRAM and to optimize matmuls (ESP-NN).
