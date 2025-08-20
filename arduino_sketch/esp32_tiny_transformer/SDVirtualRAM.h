#pragma once
#include <Arduino.h>
#include <SD.h>

class SDVirtualRAM {
public:
  static constexpr size_t PAGE_SIZE = 512;    // SD sector-aligned

  SDVirtualRAM(uint8_t csPin = SS): _csPin(csPin) {}

  bool   begin();
  // return pointer valid until next call
  const uint8_t* readPageFilePtr(const char* filename);
  // if outbuf==nullptr, return file size; else fill outbuf with file content and return bytes read
  size_t readFileToBuffer(const char* filename, char* outbuf);

  // optional no-op stub to satisfy debugger
  void syncMetadata(uint8_t unoID);

private:
  uint8_t _csPin;
  static uint8_t _pageBuf[PAGE_SIZE];
};
