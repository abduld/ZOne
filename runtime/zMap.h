
#ifndef __ZMAP_H__
#define __ZMAP_H__

struct st_zMapFunction_t {
  char * name;
  void (*f)(zMemory_t, zMemory_t);
};

struct st_zMapGroupFunction_t {
  char * name;
  void (*f)(zMemoryGroup_t, zMemoryGroup_t);
};

void zMap(zState_t st, zMapFunction_t f, zMemory_t out, zMemory_t in);
void zMap(zState_t st, zMapGroupFunction_t f, zMemoryGroup_t out, zMemoryGroup_t in);


#define zMapGroupFunction_getName(mpf)				((mpf)->name)
#define zMapGroupFunction_getFunction(mpf)			((mpf)->f)
#define zMapGroupFunction_setName(mpf, val)			(zMapGroupFunction_getName(mpf) = val)
#define zMapGroupFunction_setFunction(mpf, val)		(zMapGroupFunction_getFunction(mpf) = val)


#define zMapFunction_getName(mpf)				((mpf)->name)
#define zMapFunction_getFunction(mpf)			((mpf)->f)
#define zMapFunction_setName(mpf, val)			(zMapFunction_getName(mpf) = val)
#define zMapFunction_setFunction(mpf, val)		(zMapFunction_getFunction(mpf) = val)

zMapGroupFunction_t zMapGroupFunction_new(zState_t st, const char * name, void (*f)(zMemoryGroup_t, zMemoryGroup_t));
zMapFunction_t zMapFunction_new(zState_t st, const char * name, void (*f)(zMemory_t, zMemory_t));



void zMap(zState_t st, zMapGroupFunction_t mapFun, zMemoryGroup_t out, zMemoryGroup_t in);


#endif /* __ZMAP_H__ */
