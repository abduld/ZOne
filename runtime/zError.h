

#ifndef __ZERROR_H__
#define __ZERROR_H__

typedef enum en_zErrorCode_t {
#define zError_define(err, ...) zError_##err,
#include "zError_inc.h"
#undef zError_define
  zSuccess = zError_success
} zErrorCode_t;

#define zSuccessQ(err)                                                         \
  (zError_getCode(err) == zSuccess || zError_getCode(err) == cudaSuccess)
#define zFailQ(err) (!(zSuccessQ(err)))

struct st_zError_t {
  int line;
  zErrorCode_t code;
  const char *msg;
  const char *file;
  const char *function;
  zState_t st;
};

#define zError_getMessage(err) ((err)->msg)
#define zError_getCode(err) ((err)->code)
#define zError_getLine(err) ((err)->line)
#define zError_getFile(err) ((err)->file)
#define zError_getFunction(err) ((err)->function)
#define zError_getState(err) ((err)->st)
#define zError_getLoop(err) ((err)->loop)

#define zError_setMessage(err, val) (zError_getMessage(err) = val)
#define zError_setCode(err, val) (zError_getCode(err) = val)
#define zError_setLine(err, val) (zError_getLine(err) = val)
#define zError_setFile(err, val) (zError_getFile(err) = val)
#define zError_setFunction(err, val) (zError_getFunction(err) = val)
#define zError_setState(err, val) (zError_getState(err) = val)
#define zError_setLoop(err, val) (zError_getLoop(err) = val)

#define zError(err, code)                                                      \
  zError_update(err, zError_##code, zFile, zFunction, zLine)

extern zError_t zError_new();
extern void zError_delete(zError_t err);

extern void zError_update(zError_t err, zErrorCode_t code, const char *file,
                          const char *fun, int line);

extern char *zError_toLog(zError_t err);

#endif /* __ZERROR_H__ */
