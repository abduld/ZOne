
#ifndef __ZLOGGER_H__
#define __ZLOGGER_H__

typedef enum en_zLogLevel_t {
  zLogLevel_unknown = -1,
  zLogLevel_OFF = 0,
  zLogLevel_FATAL,
  zLogLevel_ERROR,
  zLogLevel_WARN,
  zLogLevel_INFO,
  zLogLevel_DEBUG,
  zLogLevel_TRACE
} zLogLevel_t;

struct st_zLogEntry_t {
  int line;
  char *msg;
  uint64_t time;
  const char *fun;
  const char *file;
  zLogLevel_t level;
  zLogEntry_t next;
};

struct st_zLogger_t {
  int length;
  zLogEntry_t head;
  zLogLevel_t level;
};

#define zLogEntry_getMessage(elem) ((elem)->msg)
#define zLogEntry_getTime(elem) ((elem)->time)
#define zLogEntry_getLevel(elem) ((elem)->level)
#define zLogEntry_getNext(elem) ((elem)->next)
#define zLogEntry_getLine(elem) ((elem)->line)
#define zLogEntry_getFunction(elem) ((elem)->fun)
#define zLogEntry_getFile(elem) ((elem)->file)

#define zLogEntry_setMessage(elem, val) (zLogEntry_getMessage(elem) = val)
#define zLogEntry_setTime(elem, val) (zLogEntry_getTime(elem) = val)
#define zLogEntry_setLevel(elem, val) (zLogEntry_getLevel(elem) = val)
#define zLogEntry_setNext(elem, val) (zLogEntry_getNext(elem) = val)
#define zLogEntry_setLine(elem, val) (zLogEntry_getLine(elem) = val)
#define zLogEntry_setFunction(elem, val) (zLogEntry_getFunction(elem) = val)
#define zLogEntry_setFile(elem, val) (zLogEntry_getFile(elem) = val)

#define zLogger_getLength(log) ((log)->length)
#define zLogger_getHead(log) ((log)->head)
#define zLogger_getLevel(log) ((log)->level)

#define zLogger_setLength(log, val) (zLogger_getLength(log) = val)
#define zLogger_setHead(log, val) (zLogger_getHead(log) = val)

#define zLogger_incrementLength(log) (zLogger_getLength(log)++)
#define zLogger_decrementLength(log) (zLogger_getLength(log)--)

#define zLog(level, ...)                                                       \
  zLogger_append(st, zLogLevel_##level, zString(__VA_ARGS__), zFile,           \
                 zFunction, zLine)

zLogger_t zLogger_new();

void zLogger_clear(zLogger_t logger);

void zLogger_delete(zLogger_t logger);

void zLogger_append(zState_t st, zLogLevel_t level, string msg,
                    const char *file, const char *fun, int line);

#endif /* __ZLOGGER_H__ */
