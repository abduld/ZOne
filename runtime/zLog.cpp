
#include <z.h>

zLogger_t _logger = NULL;

static inline zBool zLogEntry_hasNext(zLogEntry_t elem) {
  return zLogEntry_getNext(elem) != NULL;
}

static inline zLogEntry_t zLogEntry_new() {
  zLogEntry_t elem;

  elem = zNew(struct st_zLogEntry_t);

  zLogEntry_setMessage(elem, NULL);
  zLogEntry_setTime(elem, _hrtime());
#ifndef NDEBUG
  zLogEntry_setLevel(elem, zLogLevel_TRACE);
#else
  zLogEntry_setLevel(elem, zLogLevel_OFF);
#endif
  zLogEntry_setNext(elem, NULL);

  zLogEntry_setLine(elem, -1);
  zLogEntry_setFile(elem, NULL);
  zLogEntry_setFunction(elem, NULL);

  return elem;
}

static inline zLogEntry_t zLogEntry_initialize(zLogLevel_t level, string msg,
                                               const char *file,
                                               const char *fun, int line) {
  zLogEntry_t elem;

  elem = zLogEntry_new();

  zLogEntry_setLevel(elem, level);

  zLogEntry_setMessage(elem, zString_duplicate(msg));

  zLogEntry_setLine(elem, line);
  zLogEntry_setFile(elem, file);
  zLogEntry_setFunction(elem, fun);

  return elem;
}

static inline void zLogEntry_delete(zLogEntry_t elem) {
  if (elem != NULL) {
    if (zLogEntry_getMessage(elem) != NULL) {
      zFree(zLogEntry_getMessage(elem));
    }
    zDelete(elem);
  }
  return;
}

static inline const char *getLevelName(zLogLevel_t level) {
  switch (level) {
  case zLogLevel_unknown:
    return "Unknown";
  case zLogLevel_OFF:
    return "Off";
  case zLogLevel_FATAL:
    return "Fatal";
  case zLogLevel_ERROR:
    return "Error";
  case zLogLevel_WARN:
    return "Warn";
  case zLogLevel_INFO:
    return "Info";
  case zLogLevel_DEBUG:
    return "Debug";
  case zLogLevel_TRACE:
    return "Trace";
  }
  return NULL;
}

zLogger_t zLogger_new() {
  zLogger_t logger;

  logger = zNew(struct st_zLogger_t);

  zLogger_setLength(logger, 0);
  zLogger_setHead(logger, NULL);
#ifndef NDEBUG
  zLogger_getLevel(logger) = zLogLevel_TRACE;
#else
  zLogger_getLevel(logger) = zLogLevel_OFF;
#endif

  return logger;
}

static inline void _zLogger_setLevel(zLogger_t logger, zLogLevel_t level) {
  zLogger_getLevel(logger) = level;
}

static inline void _zLogger_setLevel(zLogLevel_t level) {
  z_init();
  _zLogger_setLevel(_logger, level);
}

#define zLogger_setLevel(level) _zLogger_setLevel(zLogLevel_##level)

void zLogger_clear(zLogger_t logger) {
  if (logger != NULL) {
    zLogEntry_t tmp;
    zLogEntry_t iter;

    iter = zLogger_getHead(logger);
    while (iter != NULL) {
      tmp = zLogEntry_getNext(iter);
      zLogEntry_delete(iter);
      iter = tmp;
    }

    zLogger_setLength(logger, 0);
    zLogger_setHead(logger, NULL);
  }
}

void zLogger_delete(zLogger_t logger) {
  if (logger != NULL) {
    zLogger_clear(logger);
    zDelete(logger);
  }
  return;
}

void zLogger_append(zState_t st, zLogLevel_t level, string msg,
                    const char *file, const char *fun, int line) {
  zLogEntry_t elem;
  zLogger_t logger = zState_getLogger(st);

  if (zLogger_getLevel(logger) < level) {
    return;
  }

  elem = zLogEntry_initialize(level, msg, file, fun, line);

  zState_mutexed(Logger, {
    if (zLogger_getHead(logger) == NULL) {
      zLogger_setHead(logger, elem);
    } else {
      zLogEntry_t prev = zLogger_getHead(logger);

      while (zLogEntry_hasNext(prev)) {
        prev = zLogEntry_getNext(prev);
      }
      zLogEntry_setNext(prev, elem);
    }
    zLogger_incrementLength(logger);

    if (level <= zLogger_getLevel(logger) && elem) {
      const char *levelName = getLevelName(level);

      fprintf(stderr, "= LOG: %s: %s (In %s:%s on line %d). =\n", levelName,
              zLogEntry_getMessage(elem), zLogEntry_getFile(elem),
              zLogEntry_getFunction(elem), zLogEntry_getLine(elem));
    }
  });

  return;
}
