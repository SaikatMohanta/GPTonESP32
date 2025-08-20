#include "sd_vram_adapter.h"
#include "SDVirtualRAM.h"

static SDVirtualRAM sdv;   // single instance

bool SDVClass::begin() { return sdv.begin(); }

size_t SDVClass::getPageSize() { return SDVirtualRAM::PAGE_SIZE; }

const uint8_t* SDVClass::readPageToBuffer(const char* filename) {
  return sdv.readPageFilePtr(filename);
}

size_t SDVClass::readFileToBuffer(const char* filename, char* outbuf) {
  return sdv.readFileToBuffer(filename, outbuf);
}

SDVClass SDV;
