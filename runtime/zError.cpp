
#include <z.h>

static const char *zErrorMessages[] = {
#define zError_define(err, msg, ...) msg,
#include "error_inc.h"
#undef zError_define
};

zError_t zError_new() {
  zError_t err;

  err = zNew(struct st_zError_t);
  zError_setState(err, NULL);
  zError_setCode(err, zSuccess);
  zError_setMessage(err, zErrorMessages[zSuccess]);
  zError_setFunction(err, NULL);
  zError_setFile(err, NULL);
  zError_setLine(err, -1);
  zError_setLoop(err, NULL);

  return err;
}

void zError_delete(zError_t err) {
  if (err != NULL) {
    zDelete(err);
  }
  return;
}

void zError_update(zError_t err, zErrorCode_t code, const char *file,
                   const char *fun, int line) {
  zState_t ctx;
  const char *msg;

  if (err == NULL) {
    return;
  }

  ctx = zError_getState(err);

  zState_lockMutex(ctx);

  if (code == zError_uv) {
    uv_loop_t *loop;
    uv_err_t uvErr;

    if (zError_getLoop(err) != NULL) {
      loop = zError_getLoop(err);
    } else if (ctx != NULL && zState_getLoop(ctx)) {
      loop = zState_getLoop(ctx);
    } else {
      return;
    }

    uvErr = uv_last_error(loop);
    if (uvErr.code == UV_OK) {
      zError_setCode(err, zSuccess);
      msg = zErrorMessages[zSuccess];
    } else {
      msg = uv_strerror(uvErr);
    }
  } else {
    zAssert(code > 0);
    zAssert(code < sizeof(zErrorMessages));
    msg = zErrorMessages[code];
  }

  zError_setCode(err, code);
  zError_setMessage(err, msg);
  zError_setFile(err, file);
  zError_setFunction(err, fun);
  zError_setLine(err, line);

  zState_unlockMutex(ctx);

  return;
}

const char *zError_message(zError_t err) {
  const char *res = NULL;
  if (err != NULL) {
    if (zError_getMessage(err) == NULL) {
      zAssert(zError_getCode(err) > 0);
      zAssert(zError_getCode(err) < sizeof(zErrorMessages));
      res = zErrorMessages[zError_getCode(err)];
    } else {
      res = zError_getMessage(err);
    }
  }
  return res;
}

#define LINE_STR_LENGTH ((CHAR_BIT * sizeof(int) - 1) / 3 + 2)

char *zError_toCString(zError_t err) {
  if (err != NULL && zFailQ(err)) {
    char lineStr[LINE_STR_LENGTH];

    zStringBuffer_t buf = zStringBuffer_new();

    zStringBuffer_append(buf, "ERROR: (");
    zStringBuffer_append(buf, zError_getMessage(err));
    zStringBuffer_append(buf, ") in ");
    zStringBuffer_append(buf, zError_getFile(err));
    zStringBuffer_append(buf, "::");
    zStringBuffer_append(buf, zError_getFunction(err));
    zStringBuffer_append(buf, " on line ");

    snprintf(lineStr, LINE_STR_LENGTH, "%d", zError_getLine(err));
    zStringBuffer_append(buf, lineStr);

    zStringBuffer_append(buf, ".");

    char *res = zStringBuffer_toCString(buf);
    zStringBuffer_deleteStructure(buf);
    return res;
  }
  return NULL;
}
