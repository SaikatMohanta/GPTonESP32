#include "SDVirtualRAMDebugger.h"

void SDVirtualRAMDebugger::printMenu() {
  Serial.println(F("SDV Debugger: [S]tatus  [P]age <id>  [R]eset meta  [H]elp"));
}

void SDVirtualRAMDebugger::printPageMeta(int idx) {
  if (idx<0 || idx>=MAX_PAGES) { Serial.println(F("Invalid page idx")); return; }
  Serial.print(F("Page ")); Serial.print(idx);
  Serial.print(F(" wc=")); Serial.print(pageMeta[idx].writeCount);
  Serial.print(F(" tag=")); Serial.print(pageMeta[idx].pageTag);
  Serial.print(F(" hash[0]=")); Serial.println(pageMeta[idx].hash[0], HEX);
}

void SDVirtualRAMDebugger::handleCommand(char cmd) {
  if (cmd=='S' || cmd=='s') {
    for (int i=0;i<10;i++) printPageMeta(i);
  }
  else if (cmd=='P' || cmd=='p') {
    Serial.println(F("Enter page id:"));
    while(!Serial.available());
    int idp = Serial.parseInt();
    printPageMeta(idp);
  }
  else if (cmd=='R' || cmd=='r') {
    for (int i=0;i<MAX_PAGES;i++) {
      pageMeta[i].writeCount = 0;
      memset(pageMeta[i].pageTag, 0, sizeof(pageMeta[i].pageTag));
      memset(pageMeta[i].hash, 0, sizeof(pageMeta[i].hash));
    }
    vram.syncMetadata(0);
    Serial.println(F("Metadata reset."));
  }
  else printMenu();
}
