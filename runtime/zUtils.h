
#ifndef __ZUTILS_H__
#define __ZUTILS_H__

#include <assert.h>
#include <zCUDA.h>
#include "tbb/tbb.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/task.h"

using namespace tbb;

#define zLine __LINE__
#define zFile __FILE__
#define zFunction __func__

#define zExit()                                                                \
  do {                                                                         \
    zAssert(0);                                                                \
    exit(1);                                                                   \
  } while (0)
#define zPrint(msg)                                                            \
  do {                                                                         \
    std::cout << msg << std::endl;                                             \
  } while (0)

#ifdef Z_CONFIG_DEBUG
#define zAssert(cond) assert(cond)
#define zAssertMessage(msg, cond)                                              \
  do {                                                                         \
    if (!(cond)) {                                                             \
      zPrint(msg);                                                             \
      zAssert(cond);                                                           \
    }                                                                          \
  } while (0)
#define zError(msg) zAssertMessage(msg, False)
#else /* WB_DEBUG */
#define zAssert(...)
#define zAssertMessage(...)
#define zError(msg) zPrint(msg)
#endif /* WB_DEBUG */

#include "zMalloc.h"

#endif /* __ZUTILS_H__ */
