
#ifndef __ZUTILS_H__
#define __ZUTILS_H__


#include <assert.h>

#define zLine __LINE__
#define zFile __FILE__
#define zFunction __func__

#define zExit()  do { zAssert(0); exit(1); } while(0)
#define zPrint(msg) do { std::cout << msg << std::endl; } while (0)

#ifdef Z_CONFIG_DEBUG
#define zAssert(cond) assert(cond)
#define zAssertMessage(msg, cond)                                             \
  do {                                                                         \
    if (!(cond)) {                                                             \
      zPrint(msg);                                                            \
      zAssert(cond);                                                          \
    }                                                                          \
  } while (0)
#define zError(msg) zAssertMessage(msg, False)
#else /* WB_DEBUG */
#define zAssert(...)
#define zAssertMessage(...)
#define zError(msg) zPrint(msg)
#endif /* WB_DEBUG */

static inline void *xMalloc(size_t sz) {
  void *mem = NULL;
  if (sz != 0) {
    mem = malloc(sz);
  }
  return mem;
}

static inline void xFree(void *mem) {
  if (mem != NULL) {
    free(mem);
  }
  return;
}

static inline void *xRealloc(void *mem, size_t sz) {
  if (mem == NULL) {
    return NULL;
  } else if (sz == 0) {
    xFree(mem);
    return NULL;
  } else {
    void *res = realloc(mem, sz);
    zAssert(res != NULL);
    return res;
  }
}

#define zNew(type) ((type *)zMalloc(sizeof(type)))
#define zNewArray(type, len) ((type *)zMalloc((len) * sizeof(type)))
#define zMalloc(sz) xMalloc(sz)
#define zDelete(var) zFree(var)
#define zFree(var)   do { xFree(var); var = NULL; } while(0)
#define wbRealloc(var, newSize) xRealloc(var, newSize)
#define wbReallocArray(t, m, n) ((t *)xRealloc(m, n * sizeof(t)))


#endif /* __ZUTILS_H__ */
