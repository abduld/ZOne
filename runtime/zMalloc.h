
#ifndef __ZMALLOC_H__
#define __ZMALLOC_H__

enum { zMalloc_fieldSize = 0, zMalloc_fieldCount };

#define zMalloc_padding 0 //*(zMalloc_fieldCount * sizeof(size_t))
#define zMalloc_address(mem) (((char *)mem) - zMalloc_padding)
#define zMalloc_getSize(                                                       \
    mem)                         //                                                  \
  //(((size_t *)zMalloc_address(mem))[zMalloc_fieldSize])
#define zMalloc_setSize(mem, sz) //(zMalloc_getSize(mem) = sz)

static inline void *xMalloc(size_t sz) {
  void *mem = NULL;
  if (sz != 0) {
    mem = malloc(sz + zMalloc_padding);
  }
  if (mem != NULL) {
    mem = ((char *)mem) + zMalloc_padding;
    zMalloc_setSize(mem, sz);
    return mem;
  } else {
    return NULL;
  }
}

static inline void xFree(void *mem) {
  if (mem != NULL) {
    free(zMalloc_address(mem));
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
#if 0
    void *tm = zMalloc_address(mem);
    void *res = realloc(tm, sz);
    if (res != NULL) {
      res = ((char *)res) + zMalloc_padding;
      zAssert(res != NULL);
      zMalloc_setSize(res, sz);
    }
#else
    void *res = realloc(mem, sz);
    if (res != NULL) {
      cudaHostRegister(res, sz, cudaHostRegisterPortable);
    }
#endif
    return res;
  }
}

static inline void *xcuMalloc(size_t sz0) {
  void *mem = NULL;
  size_t sz = sz0 + zMalloc_padding;
  if (sz != 0) {
    cudaError_t err = cudaMallocHost((void **)&mem, sz);
    if (err == cudaSuccess) {
      mem = ((char *)mem) + zMalloc_padding;
      zMalloc_setSize(mem, sz);
      return mem;
    } else {
      printf("ERRROR::: Failed to allocate %s\n", cudaGetErrorString(err));
    }
  }
  return NULL;
}

static inline void xcuFree(void *mem) {
  if (mem != NULL) {
    cudaError_t err = cudaFreeHost(zMalloc_address(mem));
    zAssert(err == cudaSuccess);
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
#if 0
    void *tm = zMalloc_address(mem);
    void *res = realloc(tm, sz);
    if (res != NULL) {
      res = ((char *)res) + zMalloc_padding;
      zAssert(res != NULL);
      zMalloc_setSize(res, sz);
    }
#else
    void *res = realloc(mem, sz);
    if (res != NULL) {
      cudaHostRegister(res, sz, cudaHostRegisterPortable);
    }
#endif
    return res;
  }
}

#define nNew(typ) ((typ *)nMalloc(sizeof(typ)))
#define zNew(typ) ((typ *)zMalloc(sizeof(typ)))
#define zNewArray(typ, len) ((typ *)zMalloc((len) * sizeof(typ)))
#define nNewArray(typ, len) ((typ *)nMalloc((len) * sizeof(typ)))
#define nMalloc(sz) xMalloc(sz)
#define zMalloc(sz) xcuMalloc(sz)
#define nDelete(var) xFree(var)
#define zDelete(var) zFree(var)
#define nFree(var)                                                             \
  do {                                                                         \
    xFree(var);                                                                \
    var = NULL;                                                                \
  } while (0)
#define zFree(var)                                                             \
  do {                                                                         \
    xcuFree(var);                                                              \
    var = NULL;                                                                \
  } while (0)
#define zRealloc(var, newSize) xcuRealloc(var, newSize)
#define nRealloc(var, newSize) xRealloc(var, newSize)
#define zReallocArray(t, m, n) ((t *)xcuRealloc(m, n * sizeof(t)))
#define nReallocArray(t, m, n) ((t *)xRealloc(m, n * sizeof(t)))

#endif /* __ZMALLOC_H__ */
