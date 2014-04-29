#include <z.h>

zFile_t zFile_new(zState_t st, const char *path, int flags) {
  zFile_t file;

  if (path == NULL) {
    return NULL;
  }

  file = zNew(struct st_zFile_t);

  zFile_setPath(file, zString_duplicate(path));
  zFile_setState(file, st);
  zFile_setFileHandle(file, -1);
  zFile_setFlags(file, flags);
  zFile_setOpenedQ(file, zFalse);

  zFile_setOffset(file, 0);

  return file;
}

void zFile_delete(zFile_t file) {
  if (file != NULL) {
    if (zFile_getPath(file)) {
      zDelete(zFile_getPath(file));
    }
    zDelete(file);
  }
  return;
}

void zFile_open(zFile_t file) { zFile_open(file, S_IREAD | S_IWRITE); }

void zFile_readChunk(zFile_t file, void *buffer, size_t sz, size_t offset) {
  zState_t st;

  st = zFile_getState(file);
  zAssert(st != NULL);

  fd = open(zFile_getPath(file), zFile_getFlags(file));
  zAssert(fd > 0);
  fseek(fd, offset, 0);
  const char *memblock = mmap(NULL, sz, PROT_WRITE, MAP_PRIVATE, fd, 0);
  zAssert(memblock != MAP_FAILED);

  memcpy(*data, memblock, sz);

  close(fd);
}

void zFile_write(zFile_t file, const char *text) {

  // TODO

  return;
}
