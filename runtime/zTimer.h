

#ifndef __ZTIMER_H__
#define __ZTIMER_H__

#include <time.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif /* __APPLE__ */

#ifdef _WIN32
extern uint64_t _hrtime_frequency;
#endif /* _WIN32 */

extern zTimer_t _timer;

struct st_zTimerNode_t {
  int id;
  int level;
  zBool stoppedQ;
  string kind;
  uint64_t startTime;
  uint64_t endTime;
  uint64_t elapsedTime;
  int startLine;
  int endLine;
  const char *startFunction;
  const char *endFunction;
  const char *startFile;
  const char *endFile;
  zTimerNode_t next;
  zTimerNode_t prev;
  zTimerNode_t parent;
  char *msg;
};

struct st_zTimer_t {
  size_t length;
  zTimerNode_t head;
  zTimerNode_t tail;
  uint64_t startTime;
  uint64_t endTime;
  uint64_t elapsedTime;
};

#define zTimerNode_getId(node) ((node)->id)
#define zTimerNode_getLevel(node) ((node)->level)
#define zTimerNode_getStoppedQ(node) ((node)->stoppedQ)
#define zTimerNode_getKind(node) ((node)->kind)
#define zTimerNode_getStartTime(node) ((node)->startTime)
#define zTimerNode_getEndTime(node) ((node)->endTime)
#define zTimerNode_getElapsedTime(node) ((node)->elapsedTime)
#define zTimerNode_getStartLine(node) ((node)->startLine)
#define zTimerNode_getEndLine(node) ((node)->endLine)
#define zTimerNode_getStartFunction(node) ((node)->startFunction)
#define zTimerNode_getEndFunction(node) ((node)->endFunction)
#define zTimerNode_getStartFile(node) ((node)->startFile)
#define zTimerNode_getEndFile(node) ((node)->endFile)
#define zTimerNode_getNext(node) ((node)->next)
#define zTimerNode_getPrevious(node) ((node)->prev)
#define zTimerNode_getParent(node) ((node)->parent)
#define zTimerNode_getMessage(node) ((node)->msg)

#define zTimerNode_setId(node, val) (zTimerNode_getId(node) = val)
#define zTimerNode_setLevel(node, val) (zTimerNode_getLevel(node) = val)
#define zTimerNode_setStoppedQ(node, val) (zTimerNode_getStoppedQ(node) = val)
#define zTimerNode_setKind(node, val) (zTimerNode_getKind(node) = val)
#define zTimerNode_setStartTime(node, val) (zTimerNode_getStartTime(node) = val)
#define zTimerNode_setEndTime(node, val) (zTimerNode_getEndTime(node) = val)
#define zTimerNode_setElapsedTime(node, val)                                   \
  (zTimerNode_getElapsedTime(node) = val)
#define zTimerNode_setStartLine(node, val) (zTimerNode_getStartLine(node) = val)
#define zTimerNode_setEndLine(node, val) (zTimerNode_getEndLine(node) = val)
#define zTimerNode_setStartFunction(node, val)                                 \
  (zTimerNode_getStartFunction(node) = val)
#define zTimerNode_setEndFunction(node, val)                                   \
  (zTimerNode_getEndFunction(node) = val)
#define zTimerNode_setStartFile(node, val) (zTimerNode_getStartFile(node) = val)
#define zTimerNode_setEndFile(node, val) (zTimerNode_getEndFile(node) = val)
#define zTimerNode_setNext(node, val) (zTimerNode_getNext(node) = val)
#define zTimerNode_setPrevious(node, val) (zTimerNode_getPrevious(node) = val)
#define zTimerNode_setParent(node, val) (zTimerNode_getParent(node) = val)
#define zTimerNode_setMessage(node, val) (zTimerNode_getMessage(node) = val)

#define zTimerNode_stoppedQ(node) (zTimerNode_getStoppedQ(node) == zTrue)
#define zTimerNode_hasNext(node) (zTimerNode_getNext(node) != NULL)
#define zTimerNode_hasPrevious(node) (zTimerNode_getPrevious(node) != NULL)
#define zTimerNode_hasParent(node) (zTimerNode_getParent(node) != NULL)

uint64_t zNow(void);

zTimer_t zTimer_new(void);
void zTimer_delete(zTimer_t timer);

string zTimer_toString(zTimer_t timer);

zTimerNode_t zTimer_start(zState_t st, zTimerKind_t kind, const char *file,
                          const char *fun, int line);
zTimerNode_t zTimer_start(zState_t st, zTimerKind_t kind, string msg,
                          const char *file, const char *fun, int line);
void zTimer_stop(zState_t st, zTimerKind_t kind, string msg, const char *file,
                 const char *fun, int line);
void zTimer_stop(zState_t st, zTimerKind_t kind, const char *file,
                 const char *fun, int line);

#define zTime_start(kind, ...)                                                 \
  zTimer_start(st, zTimerKind_##kind, zString(__VA_ARGS__), zFile, zFunction,  \
               zLine)
#define zTime_stop(kind, ...)                                                  \
  zTimer_stop(st, zTimerKind_##kind, zString(__VA_ARGS__), zFile, zFunction,   \
              zLine)

#endif /* __ZTIMER_H__ */
