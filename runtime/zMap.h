
#ifndef __ZMAP_H__
#define __ZMAP_H__

typedef struct st_zMapFunction_t {
  string name;
  zMemoryGroup_t (*f)(dim3 blockDim, dim3 gridDim, zMemoryGroup_t)
} zMapFunction_t;

typedef struct st_zMapGroupFunction_t {
  string name;
  typedef zMemoryGroup_t (*f)(dim3 blockDim, dim3 gridDim, zMemoryGroup_t);
} zMapGroupFunction_t;

void zMap(zState_t st, zMemory_t out, zMapFunction_t f, zMemory_t in);
void zMap(zState_t st, zMemoryGroup_t out, zMapGroupFunction_t f,
          zMemoryGroup_t in);

#endif /* __ZMAP_H__ */
