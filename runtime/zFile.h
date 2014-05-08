
#ifndef __ZFILE_H__
#define __ZFILE_H__

#include "z.h"

struct st_zFile_t {
  zState_t st;
  char *path;
  int flags;
};

#define zFile_getState(fs) ((fs)->st)
#define zFile_getPath(fs) ((fs)->path)
#define zFile_getFlags(fs) ((fs)->flags)
#define zFile_getOpenedQ(fs) ((fs)->openedQ)

#define zFile_setState(fs, val) (zFile_getState(fs) = val)
#define zFile_setPath(fs, val) (zFile_getPath(fs) = val)
#define zFile_setFlags(fs, val) (zFile_getFlags(fs) = val)
#define zFile_setOpenedQ(fs, val) (zFile_getOpenedQ(fs) = val)

static inline char *zEnvironment_get(const char *name) {
#ifdef _WIN32
#error "todo"
#else
  return getenv(name);
#endif
}

static inline char *zDirectory_getTemporary() {
  char *buffer;

#define _zDirectory_getTemporary(name)                                         \
  buffer = zEnvironment_get(name);                                             \
  if (buffer != NULL) {                                                        \
    return buffer;                                                             \
  }

  _zDirectory_getTemporary("TEMP");
  _zDirectory_getTemporary("TMPDIR");
  _zDirectory_getTemporary("TMP");

#undef _zDirectory_getTemporary

  return NULL;
}

#ifdef _WIN32
#define zDirectory_slash "\\"
#else
#define zDirectory_slash "/"
#endif

static inline char *zFile_nameJoin(const char *dir, const char *file) {
  string out = zString(dir, zDirectory_slash, file);
  return zString_duplicate(out);
}

static inline zBool_t zFile_existsQ(const char *path) {
  FILE *file;

  if (file = fopen(path, "r")) {
    fclose(file);
    return zTrue;
  }
  return zFalse;
}

zFile_t zFile_new(zState_t st, const char *path, int flags);
void zFile_readChunk(zFile_t file, void *buffer, size_t sz, size_t offset);
void zFile_delete(zFile_t file);
void zFile_write(zFile_t file, const void *data, size_t byteCount);

#endif /* __ZFILE_H__ */
