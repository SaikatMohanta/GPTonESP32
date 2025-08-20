#pragma once
#include <Arduino.h>
#include "SDVirtualRAM_Meta.h"
#include "SDVirtualRAM.h"

class SDVirtualRAMDebugger {
public:
  SDVirtualRAMDebugger(SDVirtualRAM& v, uint8_t id): vram(v), id(id) {}
  void handleCommand(char cmd);
  void printMenu();
  void printPageMeta(int idx);
private:
  SDVirtualRAM& vram;
  uint8_t id;
};
