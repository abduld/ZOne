

#include <z.h>
#include <time.h>

#ifdef _WIN32
uint64_t zNow_frequency = 0;
#endif /* _WIN32 */

#ifdef __APPLE__
static double o_timebase = 0;
static uint64_t o_timestart = 0;
#endif /* __APPLE__ */

uint64_t zNow(void) {
#define NANOSEC ((uint64_t)1e9)
#ifdef _MSC_VER
  LARGE_INTEGER counter;
  if (!QueryPerformanceCounter(&counter)) {
    return 0;
  }
  return ((uint64_t)counter.LowPart * NANOSEC / zNow_frequency) +
         (((uint64_t)counter.HighPart * NANOSEC / zNow_frequency) << 32);
#else
  struct timespec ts;
#ifdef __APPLE__
#define O_NANOSEC (+1.0E-9)
#define O_GIGA UINT64_C(1000000000)
  if (!o_timestart) {
    mach_timebase_info_data_t tb = { 0 };
    mach_timebase_info(&tb);
    o_timebase = tb.numer;
    o_timebase /= tb.denom;
    o_timestart = mach_absolute_time();
  }
  double diff = (mach_absolute_time() - o_timestart) * o_timebase;
  ts.tv_sec = diff * O_NANOSEC;
  ts.tv_nsec = diff - (ts.tv_sec * O_GIGA);
#undef O_NANOSEC
#undef O_GIGA
#else  /* __APPLE__ */
  clock_gettime(CLOCK_MONOTONIC, &ts);
#endif /* __APPLE__ */
  return (((uint64_t)ts.tv_sec) * NANOSEC + ts.tv_nsec);
#endif /* _MSC_VER */
#undef NANOSEC
}

static inline uint64_t getTime(void) {
#if Z_CONFIG_SYNC_CUDA_TIME
  cudaThreadSynchronize();
#endif /* Z_CONFIG_SYNC_CUDA_TIME */

  return zNow();
}

static inline zTimerNode_t zTimerNode_new(int id, string kind,
                                          const char *file, const char *fun,
                                          int startLine) {
  zTimerNode_t node = zNew(struct st_zTimerNode_t);
  zTimerNode_setId(node, id);
  zTimerNode_setLevel(node, 0);
  zTimerNode_setStoppedQ(node, zFalse);
  zTimerNode_setKind(node, kind);
  zTimerNode_setStartTime(node, 0);
  zTimerNode_setEndTime(node, 0);
  zTimerNode_setElapsedTime(node, 0);
  zTimerNode_setStartLine(node, startLine);
  zTimerNode_setEndLine(node, 0);
  zTimerNode_setStartFunction(node, fun);
  zTimerNode_setEndFunction(node, NULL);
  zTimerNode_setStartFile(node, file);
  zTimerNode_setEndFile(node, NULL);
  zTimerNode_setNext(node, NULL);
  zTimerNode_setPrevious(node, NULL);
  zTimerNode_setParent(node, NULL);
  zTimerNode_setMessage(node, NULL);
  return node;
}

static inline void zTimerNode_delete(zTimerNode_t node) {
  if (node != NULL) {
    if (zTimerNode_getMessage(node)) {
      zDelete(zTimerNode_getMessage(node));
    }
    zDelete(node);
  }
}

static inline string zTimerNode_toJSON(zTimerNode_t node) {
  if (node == NULL) {
    return "";
  } else {
    stringstream ss;

    ss << "{\n";
    ss << zString_quote("id") << ":" << zTimerNode_getId(node) << ",\n";
    ss << zString_quote("stopped") << ":"
       << zString(zTimerNode_stoppedQ(node) ? "true" : "false") << ",\n";
    ss << zString_quote("kind") << ":"
       << zString_quote(zTimerNode_getKind(node)) << ",\n";
    ss << zString_quote("start_time") << ":" << zTimerNode_getStartTime(node)
       << ",\n";
    ss << zString_quote("end_time") << ":" << zTimerNode_getEndTime(node)
       << ",\n";
    ss << zString_quote("elapsed_time") << ":"
       << zTimerNode_getElapsedTime(node) << ",\n";
    ss << zString_quote("start_line") << ":" << zTimerNode_getStartLine(node)
       << ",\n";
    ss << zString_quote("end_line") << ":" << zTimerNode_getEndLine(node)
       << ",\n";
    ss << zString_quote("start_function") << ":"
       << zString_quote(zTimerNode_getStartFunction(node)) << ",\n";
    ss << zString_quote("end_function") << ":"
       << zString_quote(zTimerNode_getEndFunction(node)) << ",\n";
    ss << zString_quote("start_file") << ":"
       << zString_quote(zTimerNode_getStartFile(node)) << ",\n";
    ss << zString_quote("end_file") << ":"
       << zString_quote(zTimerNode_getEndFile(node)) << ",\n";
    ss << zString_quote("parent_id") << ":"
       << zString(zTimerNode_hasParent(node)
                      ? zTimerNode_getId(zTimerNode_getParent(node))
                      : -1) << ",\n";
    ss << zString_quote("message") << ":"
       << zString_quote(zTimerNode_getMessage(node)) << "\n";
    ss << "}";

    return ss.str();
  }
}

#define zTimer_getLength(timer) ((timer)->length)
#define zTimer_getHead(timer) ((timer)->head)
#define zTimer_getTail(timer) ((timer)->tail)
#define zTimer_getStartTime(timer) ((timer)->startTime)
#define zTimer_getEndTime(timer) ((timer)->endTime)
#define zTimer_getElapsedTime(timer) ((timer)->elapsedTime)

#define zTimer_setLength(timer, val) (zTimer_getLength(timer) = val)
#define zTimer_setHead(timer, val) (zTimer_getHead(timer) = val)
#define zTimer_setTail(timer, val) (zTimer_getTail(timer) = val)
#define zTimer_setStartTime(node, val) (zTimer_getStartTime(node) = val)
#define zTimer_setEndTime(node, val) (zTimer_getEndTime(node) = val)
#define zTimer_setElapsedTime(node, val) (zTimer_getElapsedTime(node) = val)

#define zTimer_incrementLength(timer) (zTimer_getLength(timer)++)
#define zTimer_decrementLength(timer) (zTimer_getLength(timer)--)

#define zTimer_emptyQ(timer) (zTimer_getLength(timer) == 0)

void zTimer_delete(zTimer_t timer) {
  if (timer != NULL) {
    zTimerNode_t tmp, iter;

    iter = zTimer_getHead(timer);
    while (iter) {
      tmp = zTimerNode_getNext(iter);
      zTimerNode_delete(iter);
      iter = tmp;
    }

    zDelete(timer);
  }
}

static string zTimer_toJSON(zTimer_t timer) {
  if (timer == NULL) {
    return NULL;
  } else {
    stringstream ss;
    zTimerNode_t iter;
    uint64_t currentTime;

    currentTime = getTime();

    zTimer_setEndTime(timer, currentTime);
    zTimer_setElapsedTime(timer, currentTime - zTimer_getStartTime(timer));

    ss << "{\n";
    ss << zString_quote("start_time") << ":" << zTimer_getStartTime(timer)
       << ",\n";
    ss << zString_quote("end_time") << ":" << zTimer_getEndTime(timer) << ",\n";
    ss << zString_quote("elapsed_time") << ":" << zTimer_getElapsedTime(timer)
       << ",\n";
    ss << zString_quote("elements") << ":[\n";
    for (iter = zTimer_getHead(timer); iter != NULL;
         iter = zTimerNode_getNext(iter)) {
      if (!zTimerNode_stoppedQ(iter)) {
        zTimerNode_setEndTime(iter, currentTime);
        zTimerNode_setElapsedTime(iter,
                                  currentTime - zTimerNode_getStartTime(iter));
      }
      ss << zTimerNode_toJSON(iter);
      if (zTimerNode_getNext(iter) != NULL) {
        ss << ",\n";
      }
    }
    ss << "]\n";
    ss << "}";

    return ss.str();
  }
}

static string zTimer_toCString(zTimer_t timer) {
  return zString_duplicate(zTimer_toJSON(timer));
}

zTimer_t zTimer_new(void) {
  zTimer_t timer = zNew(struct st_zTimer_t);
  zTimer_setLength(timer, 0);
  zTimer_setHead(timer, NULL);
  zTimer_setTail(timer, NULL);
  zTimer_setStartTime(timer, getTime());
  zTimer_setEndTime(timer, 0);
  zTimer_setElapsedTime(timer, 0);

  return timer;
}

static inline zTimerNode_t _findParent(zTimer_t timer) {
  zTimerNode_t iter;

  for (iter = zTimer_getTail(timer); iter != NULL;
       iter = zTimerNode_getPrevious(iter)) {
    if (!zTimerNode_stoppedQ(iter)) {
      return iter;
    }
  }
  return NULL;
}

static inline void _insertIntoList(zTimer_t timer, zTimerNode_t node) {
  if (zTimer_emptyQ(timer)) {
    zTimer_setHead(timer, node);
    zTimer_setTail(timer, node);
  } else {
    zTimerNode_t end = zTimer_getTail(timer);
    zTimer_setTail(timer, node);
    zTimerNode_setNext(end, node);
    zTimerNode_setPrevious(node, end);
  }
  zTimer_incrementLength(timer);
}

zTimerNode_t zTimer_start(zState_t st, string kind, const char *file,
                          const char *fun, int line) {
  int id;
  uint64_t currentTime;
  zTimerNode_t node;
  zTimerNode_t parent;

  zTimer_t timer = zState_getTimer(st);

  zState_mutexed(Timer, {

    currentTime = getTime();

    id = zTimer_getLength(timer);

    node = zTimerNode_new(id, kind, file, fun, line);

    parent = _findParent(timer);
    _insertIntoList(timer, node);

    zTimerNode_setStartTime(node, currentTime);
    zTimerNode_setParent(node, parent);
    if (parent != NULL) {
      zTimerNode_setLevel(node, zTimerNode_getLevel(parent) + 1);
    }
  });

  return node;
}

zTimerNode_t zTimer_start(zState_t st, string kind, string msg,
                          const char *file, const char *fun, int line) {
  zTimerNode_t node = zTimer_start(st, kind, file, fun, line);
  zTimerNode_setMessage(node, zString_duplicate(msg));
  return node;
}

static inline zTimerNode_t _findNode(zTimer_t timer, string kind,
                                     string msg) {
  zTimerNode_t iter;

  for (iter = zTimer_getTail(timer); iter != NULL;
       iter = zTimerNode_getPrevious(iter)) {
    if (msg == "") {
      if (!zTimerNode_stoppedQ(iter) && zString_sameQ(zTimerNode_getKind(iter), kind)) {
        return iter;
      }
    } else {
      if (!zTimerNode_stoppedQ(iter) && zString_sameQ(zTimerNode_getKind(iter), kind) &&
          msg == zTimerNode_getMessage(iter)) {
        return iter;
      }
    }
  }
  return NULL;
}

void zTimer_stop(zState_t st, string kind, string msg, const char *file,
                 const char *fun, int line) {
  uint64_t currentTime;
  zTimerNode_t node;

  zTimer_t timer = zState_getTimer(st);

  zState_mutexed(Timer, {
    currentTime = getTime();

    node = _findNode(timer, kind, msg);

    zAssert(node != NULL);
    if (node == NULL) {
      return;
    }

    zTimerNode_setEndTime(node, currentTime);
    zTimerNode_setElapsedTime(node,
                              currentTime - zTimerNode_getStartTime(node));
    zTimerNode_setEndLine(node, line);
    zTimerNode_setEndFunction(node, fun);
    zTimerNode_setEndFile(node, file);
    zTimerNode_setStoppedQ(node, zTrue);
  });

  return;
}

void zTimer_stop(zState_t st, string kind, const char *file,
                 const char *fun, int line) {
  zTimer_stop(st, kind, "", file, fun, line);
}
