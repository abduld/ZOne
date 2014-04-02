
#ifndef __ZFILE_H__
#define __ZFILE_H__

#include "z.h"

struct st_zFile_t {
  zContext_t ctx;
  char *path;
  int flags;
  uv_file fh;
  uv_fs_t openRequest;
  uv_fs_t readRequest;
  uv_fs_t writeRequest;
  zBool openedQ;
  size_t offset;
};

#define zFile_getContext(fs) ((fs)->ctx)
#define zFile_getPath(fs) ((fs)->path)
#define zFile_getFlags(fs) ((fs)->flags)
#define zFile_getFileHandle(fs) ((fs)->fh)
#define zFile_getOpenRequest(fs) ((fs)->openRequest)
#define zFile_getReadRequest(fs) ((fs)->readRequest)
#define zFile_getWriteRequest(fs) ((fs)->writeRequest)
#define zFile_getOpenedQ(fs) ((fs)->openedQ)
#define zFile_getOffset(fs) ((fs)->offset)

#define zFile_setContext(fs, val) (zFile_getContext(fs) = val)
#define zFile_setPath(fs, val) (zFile_getPath(fs) = val)
#define zFile_setFlags(fs, val) (zFile_getFlags(fs) = val)
#define zFile_setFileHandle(fs, val) (zFile_getFileHandle(fs) = val)
#define zFile_setOpenRequest(fs, val) (zFile_getOpenRequest(fs) = val)
#define zFile_setReadRequest(fs, val) (zFile_getReadRequest(fs) = val)
#define zFile_setWriteRequest(fs, val) (zFile_getWriteRequest(fs) = val)
#define zFile_setOpenedQ(fs, val) (zFile_getOpenedQ(fs) = val)
#define zFile_setOffset(fs, val) (zFile_getOffset(fs) = val)

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
  string out = zString(dim, zDirectory_slash, file);
  return zString_duplicate(out);
}

static inline zBool zFile_existsQ(const char *path) {
  FILE *file;

  if (file = fopen(path, "r")) {
    fclose(file);
    return zTrue;
  }
  return zFalse;
}

zFile_t zFile_new(const char *path, int flags);
void zFile_delete(zFile_t file);
void zFile_open(zFile_t file);
void zFile_close(zFile_t file);
void zFile_write(zFile_t file, const char *text);

#endif /* __ZFILE_H__ */
