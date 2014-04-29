
#ifndef __ZUTILS_H__
#define __ZUTILS_H__

#include <assert.h>
#include <cmath>
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
#define zPrint_error(msg) zAssertMessage(msg, False)
#else /* WB_DEBUG */
#define zAssert(...)
#define zAssertMessage(...)
#define zPrint_error(msg) zPrint(msg)
#endif /* WB_DEBUG */

#include "zMalloc.h"


template <typename T0, typename T1>
static T0 zCeil(const T0 & n, const T1 & d) {
  return (T0) ceil(static_cast<double>(n) / static_cast<double>(d));
}

template <>
int zCeil(const int & n, const int & d) {
  return (n + (d - 1)) / d;
}

template <>
size_t zCeil(const size_t & n, const size_t & d) {
  return (n + (d - 1)) / d;
}

template <>
size_t zCeil(const size_t & n, const int & d) {
  return (n + (d - 1)) / d;
}

template <typename T>
static T zMin(const T & m, const T & n) {
  return m < n ? m : n;
}

template <typename T>
static T zMax(const T & m, const T & n) {
  return m > n ? m : n;
}


#endif /* __ZUTILS_H__ */
