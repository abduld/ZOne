
#ifndef __ZMAP_H__
#define __ZMAP_H__

struct st_zMapFunction_t {
  char *name;
  void (*f1)(zMemory_t, zMemory_t);
  void (*f2)(zMemory_t, zMemory_t, zMemory_t);
  void (*f3)(zMemory_t, zMemory_t, zMemory_t, zMemory_t);
  void (*f4)(zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t);
  void (*f5)(zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t);
  void (*f6)(zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t,
             zMemory_t);
  void (*f7)(zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t,
             zMemory_t, zMemory_t);
  void (*f8)(zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t,
             zMemory_t, zMemory_t, zMemory_t);
};

struct st_zMapGroupFunction_t {
  char *name;
  void (*f1)(zMemory_t, zMemory_t);
  void (*f2)(zMemory_t, zMemory_t, zMemory_t);
  void (*f3)(zMemory_t, zMemory_t, zMemory_t, zMemory_t);
  void (*f4)(zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t);
  void (*f5)(zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t);
  void (*f6)(zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t,
             zMemory_t);
  void (*f7)(zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t,
             zMemory_t, zMemory_t);
  void (*f8)(zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t, zMemory_t,
             zMemory_t, zMemory_t, zMemory_t);
};

void zMap(zState_t st, zMapFunction_t f, zMemory_t out, zMemory_t in);
void zMap(zState_t st, zMapFunction_t f, zMemory_t out, zMemory_t in0,
          zMemory_t in1);
void zMap(zState_t st, zMapFunction_t f, zMemory_t out, zMemory_t in0,
          zMemory_t in1, zMemory_t in2);
void zMap(zState_t st, zMapFunction_t f, zMemory_t out, zMemory_t in0,
          zMemory_t in1, zMemory_t in2, zMemory_t in3);
void zMap(zState_t st, zMapFunction_t f, zMemory_t out, zMemory_t in0,
          zMemory_t in1, zMemory_t in2, zMemory_t in3, zMemory_t in4);
void zMap(zState_t st, zMapFunction_t f, zMemory_t out, zMemory_t in0,
          zMemory_t in1, zMemory_t in2, zMemory_t in3, zMemory_t in4,
          zMemory_t in5);
void zMap(zState_t st, zMapFunction_t f, zMemory_t out, zMemory_t in0,
          zMemory_t in1, zMemory_t in2, zMemory_t in3, zMemory_t in4,
          zMemory_t in5, zMemory_t in6);
void zMap(zState_t st, zMapFunction_t f, zMemory_t out, zMemory_t in0,
          zMemory_t in1, zMemory_t in2, zMemory_t in3, zMemory_t in4,
          zMemory_t in5, zMemory_t in6, zMemory_t in7);
void zMap(zState_t st, zMapFunction_t f, zMemory_t out, zMemory_t in0,
          zMemory_t in1, zMemory_t in2, zMemory_t in3, zMemory_t in4,
          zMemory_t in5, zMemory_t in6, zMemory_t in7, zMemory_t in8);

void zMap(zState_t st, zMapGroupFunction_t f, zMemoryGroup_t out,
          zMemoryGroup_t in);
void zMap(zState_t st, zMapGroupFunction_t f, zMemoryGroup_t out,
          zMemoryGroup_t in0, zMemoryGroup_t in1);
void zMap(zState_t st, zMapGroupFunction_t f, zMemoryGroup_t out,
          zMemoryGroup_t in0, zMemoryGroup_t in1, zMemoryGroup_t in2);
void zMap(zState_t st, zMapGroupFunction_t f, zMemoryGroup_t out,
          zMemoryGroup_t in0, zMemoryGroup_t in1, zMemoryGroup_t in2,
          zMemoryGroup_t in3);
void zMap(zState_t st, zMapGroupFunction_t f, zMemoryGroup_t out,
          zMemoryGroup_t in0, zMemoryGroup_t in1, zMemoryGroup_t in2,
          zMemoryGroup_t in3, zMemoryGroup_t in4);
void zMap(zState_t st, zMapGroupFunction_t f, zMemoryGroup_t out,
          zMemoryGroup_t in0, zMemoryGroup_t in1, zMemoryGroup_t in2,
          zMemoryGroup_t in3, zMemoryGroup_t in4, zMemoryGroup_t in5);
void zMap(zState_t st, zMapGroupFunction_t f, zMemoryGroup_t out,
          zMemoryGroup_t in0, zMemoryGroup_t in1, zMemoryGroup_t in2,
          zMemoryGroup_t in3, zMemoryGroup_t in4, zMemoryGroup_t in5,
          zMemoryGroup_t in6);
void zMap(zState_t st, zMapGroupFunction_t f, zMemoryGroup_t out,
          zMemoryGroup_t in0, zMemoryGroup_t in1, zMemoryGroup_t in2,
          zMemoryGroup_t in3, zMemoryGroup_t in4, zMemoryGroup_t in5,
          zMemoryGroup_t in6, zMemoryGroup_t in7);
void zMap(zState_t st, zMapGroupFunction_t f, zMemoryGroup_t out,
          zMemoryGroup_t in0, zMemoryGroup_t in1, zMemoryGroup_t in2,
          zMemoryGroup_t in3, zMemoryGroup_t in4, zMemoryGroup_t in5,
          zMemoryGroup_t in6, zMemoryGroup_t in7, zMemoryGroup_t in8);

#define zMapGroupFunction_getName(mpf) ((mpf)->name)
#define zMapGroupFunction_setName(mpf, val)                                    \
  (zMapGroupFunction_getName(mpf) = val)
#define zMapGroupFunction_getFunction1(mpf) ((mpf)->f1)
#define zMapGroupFunction_getFunction2(mpf) ((mpf)->f2)
#define zMapGroupFunction_getFunction3(mpf) ((mpf)->f3)
#define zMapGroupFunction_getFunction4(mpf) ((mpf)->f4)
#define zMapGroupFunction_getFunction5(mpf) ((mpf)->f5)
#define zMapGroupFunction_getFunction6(mpf) ((mpf)->f6)
#define zMapGroupFunction_getFunction7(mpf) ((mpf)->f7)
#define zMapGroupFunction_setFunction1(mpf, val)                               \
  (zMapGroupFunction_getFunction1(mpf) = val)
#define zMapGroupFunction_setFunction2(mpf, val)                               \
  (zMapGroupFunction_getFunction2(mpf) = val)
#define zMapGroupFunction_setFunction3(mpf, val)                               \
  (zMapGroupFunction_getFunction3(mpf) = val)
#define zMapGroupFunction_setFunction4(mpf, val)                               \
  (zMapGroupFunction_getFunction4(mpf) = val)
#define zMapGroupFunction_setFunction5(mpf, val)                               \
  (zMapGroupFunction_getFunction5(mpf) = val)
#define zMapGroupFunction_setFunction6(mpf, val)                               \
  (zMapGroupFunction_getFunction6(mpf) = val)
#define zMapGroupFunction_setFunction7(mpf, val)                               \
  (zMapGroupFunction_getFunction7(mpf) = val)

#define zMapFunction_getName(mpf) ((mpf)->name)
#define zMapFunction_setName(mpf, val) (zMapFunction_getName(mpf) = val)
#define zMapFunction_getFunction1(mpf) ((mpf)->f1)
#define zMapFunction_getFunction2(mpf) ((mpf)->f2)
#define zMapFunction_getFunction3(mpf) ((mpf)->f3)
#define zMapFunction_getFunction4(mpf) ((mpf)->f4)
#define zMapFunction_getFunction5(mpf) ((mpf)->f5)
#define zMapFunction_getFunction6(mpf) ((mpf)->f6)
#define zMapFunction_getFunction7(mpf) ((mpf)->f7)
#define zMapFunction_setFunction1(mpf, val)                                    \
  (zMapFunction_getFunction1(mpf) = val)
#define zMapFunction_setFunction2(mpf, val)                                    \
  (zMapFunction_getFunction2(mpf) = val)
#define zMapFunction_setFunction3(mpf, val)                                    \
  (zMapFunction_getFunction3(mpf) = val)
#define zMapFunction_setFunction4(mpf, val)                                    \
  (zMapFunction_getFunction4(mpf) = val)
#define zMapFunction_setFunction5(mpf, val)                                    \
  (zMapFunction_getFunction5(mpf) = val)
#define zMapFunction_setFunction6(mpf, val)                                    \
  (zMapFunction_getFunction6(mpf) = val)
#define zMapFunction_setFunction7(mpf, val)                                    \
  (zMapFunction_getFunction7(mpf) = val)

zMapGroupFunction_t zMapGroupFunction_new(zState_t st, const char *name,
                                          void (*f1)(zMemory_t, zMemory_t));
zMapGroupFunction_t zMapGroupFunction_new(zState_t st, const char *name,
                                          void (*f2)(zMemory_t, zMemory_t,
                                                     zMemory_t));
zMapGroupFunction_t zMapGroupFunction_new(zState_t st, const char *name,
                                          void (*f3)(zMemory_t, zMemory_t,
                                                     zMemory_t, zMemory_t));
zMapGroupFunction_t zMapGroupFunction_new(zState_t st, const char *name,
                                          void (*f4)(zMemory_t, zMemory_t,
                                                     zMemory_t, zMemory_t,
                                                     zMemory_t));
zMapGroupFunction_t zMapGroupFunction_new(zState_t st, const char *name,
                                          void (*f5)(zMemory_t, zMemory_t,
                                                     zMemory_t, zMemory_t,
                                                     zMemory_t, zMemory_t));
zMapGroupFunction_t
zMapGroupFunction_new(zState_t st, const char *name,
                      void (*f6)(zMemory_t, zMemory_t, zMemory_t, zMemory_t,
                                 zMemory_t, zMemory_t, zMemory_t));
zMapGroupFunction_t
zMapGroupFunction_new(zState_t st, const char *name,
                      void (*f7)(zMemory_t, zMemory_t, zMemory_t, zMemory_t,
                                 zMemory_t, zMemory_t, zMemory_t, zMemory_t));

#endif /* __ZMAP_H__ */
