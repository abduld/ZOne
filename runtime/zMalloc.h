
#ifndef __ZMALLOC_H__
#define __ZMALLOC_H__

enum {
  zMalloc_fieldSize = 0,
  zMalloc_fieldCount
};

#define zMalloc_padding (zMalloc_fieldCount * sizeof(size_t))
#define zMalloc_address(mem) (((char *) mem) - zMalloc_padding)
#define zMalloc_getSize(mem) (((size_t *) zMalloc_address(mem))[zMalloc_fieldSize])
#define zMalloc_setSize(mem, sz) (zMalloc_getSize(mem) = sz)

static inline void *xMalloc(size_t sz) {
  void *mem = NULL;
  if (sz != 0) {
    mem = malloc(sz + zMalloc_padding);
  }
  if (mem != NULL) {
    mem = ((char *) mem) + zMalloc_padding;
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
    void *tm = zMalloc_address(mem);
    void *res = realloc(tm, sz);
    if (res != NULL) {
      res = ((char *) res) + zMalloc_padding;
      zAssert(res != NULL);
      zMalloc_setSize(res, sz);
    }
    return res;
  }
}

static inline void *xcuMalloc(size_t sz) {
  void *mem = NULL;
  if (sz != 0) {
    cudaError_t err = cudaMallocHost((void **)&mem, sz + zMalloc_padding);
    if (err == cudaSuccess) {
      mem = ((char *) mem) + zMalloc_padding;
      zMalloc_setSize(mem, sz);
      return mem;
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
    void *res = xcuMalloc(sz);
    zAssert(res != NULL);
    if (res != NULL) {
      cudaError_t err =
          cudaMemcpy(res, mem, zMalloc_getSize(mem), cudaMemcpyHostToHost);
      zAssert(err == cudaSuccess);
    }
    xcuFree(mem);
    return res;
  }
}

#define zNew(typ) ((typ *)zMalloc(sizeof(typ)))
#define zNewArray(typ, len) ((typ *)zMalloc((len) * sizeof(typ)))
#define zMalloc(sz) xcuMalloc(sz)
#define zDelete(var) zFree(var)
#define zFree(var)                                                             \
  do {                                                                         \
    xcuFree(var);                                                              \
    var = NULL;                                                                \
  } while (0)
#define zRealloc(var, newSize) xcuRealloc(var, newSize)
#define zReallocArray(t, m, n) ((t *)xcuRealloc(m, n * sizeof(t)))

#endif /* __ZMALLOC_H__ */
