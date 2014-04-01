
#ifndef __ZUTILS_H__
#define __ZUTILS_H__


#include <assert.h>
#include <zCUDA.h>

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
    mem = malloc(sz + sizeof(size_t));
  }
  if (mem != NULL) {
    ((size_t) mem)[0] = sz;
    return mem + sizeof(size_t);
  } else {
    return NULL;
  }
}

static inline void xFree(void *mem) {
  if (mem != NULL) {
    free(mem - sizeof(size_t));
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
    void *tm = mem - sizeof(size_t);
    void *res = realloc(tm, sz);
    if (res != NULL) {
      ((size_t) res)[0] = sz;
      zAssert(res != NULL);
    }
    return res;
  }
}

static inline void *xcuMalloc(size_t sz) {
  void *mem = NULL;
  if (sz != 0) {
    cudaError_t err = cudaMallocHost((void **) &mem, sz + sizeof(size_t));
    if (zSuccessQ(err)) {
      ((size_t) mem)[0] = sz;
      return mem + sizeof(size_t);
    }
  }
  return NULL;
}

static inline void xcuFree(void *mem) {
  if (mem != NULL) {
    free(mem - sizeof(size_t));
  }
  return;
}

static inline void *xcuRealloc(void *mem, size_t sz) {
  if (mem == NULL) {
    return NULL;
  } else if (sz == 0) {
    xFree(mem);
    return NULL;
  } else {
    void *res = xcuMalloc(sz);
    zAssert(res != NULL);
    if (res != NULL) {
      cudaError_t err = cudaMemcpy(res, mem, cudaMemcpyHostToHost);
      checkSuccess(err);
    }
    xcuFree(mem);
    return res;
  }
}

#define zNew(type) ((type *)zMalloc(sizeof(type)))
#define zNewArray(type, len) ((type *)zMalloc((len) * sizeof(type)))
#define zMalloc(sz) xcuMalloc(sz)
#define zDelete(var) zFree(var)
#define zFree(var)   do { xcuFree(var); var = NULL; } while(0)
#define zRealloc(var, newSize) xcuRealloc(var, newSize)
#define zReallocArray(t, m, n) ((t *)xcuRealloc(m, n * sizeof(t)))


#endif /* __ZUTILS_H__ */
