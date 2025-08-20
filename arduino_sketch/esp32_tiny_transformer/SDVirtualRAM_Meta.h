#ifndef SDVIRTUALRAM_META_H
#define SDVIRTUALRAM_META_H

#define MAX_PAGES 1024

#include <cstdint>

struct PageMetadata {
  uint16_t writeCount;
  char     pageTag[8];
  uint8_t  hash[20];
};

extern PageMetadata pageMeta[MAX_PAGES];

#endif
