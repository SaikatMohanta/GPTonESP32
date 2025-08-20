#pragma once
#include <Arduino.h>

class SDVClass {
public:
  bool begin();
  size_t getPageSize();
  const uint8_t* readPageToBuffer(const char* filename);
  size_t readFileToBuffer(const char* filename, char* outbuf);
};

extern SDVClass SDV;
