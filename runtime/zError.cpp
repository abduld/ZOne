
#include <z.h>

static const char *zErrorMessages[] = {
#define zError_define(err, msg, ...) msg,
#include "zError_inc.h"
#undef zError_define
};

zError_t zError_new() {
  zError_t err;

  err = nNew(struct st_zError_t);
  zError_setState(err, NULL);
  zError_setCode(err, zSuccess);
  zError_setMessage(err, zErrorMessages[zSuccess]);
  zError_setFunction(err, NULL);
  zError_setFile(err, NULL);
  zError_setLine(err, -1);

  return err;
}

void zError_delete(zError_t err) {
  if (err != NULL) {
    nDelete(err);
  }
  return;
}

void zError_update(zError_t err, zErrorCode_t code, const char *file,
                   const char *fun, int line) {
  zState_t st;
  const char *msg;

  if (err == NULL) {
    return;
  }

  st = zError_getState(err);
  if (st == NULL) {
    return;
  }

  zState_mutexed(Error, {

    zAssert(code > 0);
    zAssert(code < sizeof(zErrorMessages));
    msg = zErrorMessages[code];

    zError_setCode(err, code);
    zError_setMessage(err, msg);
    zError_setFile(err, file);
    zError_setFunction(err, fun);
    zError_setLine(err, line);
  });

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
