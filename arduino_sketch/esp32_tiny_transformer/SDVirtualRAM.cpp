#include "SDVirtualRAM.h"
#include <algorithm>  // for std::min

uint8_t SDVirtualRAM::_pageBuf[SDVirtualRAM::PAGE_SIZE];

bool SDVirtualRAM::begin() {
  if (!SD.begin(_csPin)) {
    Serial.println(F("[SDVirtualRAM] SD.begin failed"));
    return false;
  }
  return true;
}

const uint8_t* SDVirtualRAM::readPageFilePtr(const char* filename) {
  File f = SD.open(filename, FILE_READ);
  if (!f) { Serial.print(F("[SDV] open fail: ")); Serial.println(filename); return nullptr; }
  size_t rd = f.read(_pageBuf, PAGE_SIZE);
  f.close();
  if (rd < PAGE_SIZE) {
    // zero tail to keep deterministic scale read at page-end
    memset(_pageBuf+rd, 0, PAGE_SIZE-rd);
  }
  return _pageBuf;
}

size_t SDVirtualRAM::readFileToBuffer(const char* filename, char* outbuf) {
  File f = SD.open(filename, FILE_READ);
  if (!f) return 0;
  size_t sz = f.size();
  if (!outbuf) { f.close(); return sz; }
  size_t rd = f.readBytes(outbuf, sz);
  f.close();
  return rd;
}

void SDVirtualRAM::syncMetadata(uint8_t /*unoID*/) {
  // stub – extend if you implement per-page metadata persistence
}
