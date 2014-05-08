
#include "z.h"

zMapGroupFunction_t zMapGroupFunction_new(zState_t st, const char *name,
                                          void (*f)(zMemory_t, zMemory_t)) {
  zMapGroupFunction_t mpf = nNew(struct st_zMapGroupFunction_t);
  zMapGroupFunction_setName(mpf, zString_duplicate(name));
  zMapGroupFunction_setFunction1(mpf, f);
  return mpf;
}

zMapGroupFunction_t zMapGroupFunction_new(zState_t st, const char *name,
                                          void (*f)(zMemory_t, zMemory_t,
                                                    zMemory_t)) {
  zMapGroupFunction_t mpf = nNew(struct st_zMapGroupFunction_t);
  zMapGroupFunction_setName(mpf, zString_duplicate(name));
  zMapGroupFunction_setFunction2(mpf, f);
  return mpf;
}

zMapGroupFunction_t zMapGroupFunction_new(zState_t st, const char *name,
                                          void (*f)(zMemory_t, zMemory_t,
                                                    zMemory_t, zMemory_t)) {
  zMapGroupFunction_t mpf = nNew(struct st_zMapGroupFunction_t);
  zMapGroupFunction_setName(mpf, zString_duplicate(name));
  zMapGroupFunction_setFunction3(mpf, f);
  return mpf;
}

zMapGroupFunction_t zMapGroupFunction_new(zState_t st, const char *name,
                                          void (*f)(zMemory_t, zMemory_t,
                                                    zMemory_t, zMemory_t,
                                                    zMemory_t)) {
  zMapGroupFunction_t mpf = nNew(struct st_zMapGroupFunction_t);
  zMapGroupFunction_setName(mpf, zString_duplicate(name));
  zMapGroupFunction_setFunction4(mpf, f);
  return mpf;
}

zMapGroupFunction_t zMapGroupFunction_new(zState_t st, const char *name,
                                          void (*f)(zMemory_t, zMemory_t,
                                                    zMemory_t, zMemory_t,
                                                    zMemory_t, zMemory_t)) {
  zMapGroupFunction_t mpf = nNew(struct st_zMapGroupFunction_t);
  zMapGroupFunction_setName(mpf, zString_duplicate(name));
  zMapGroupFunction_setFunction5(mpf, f);
  return mpf;
}

zMapGroupFunction_t
zMapGroupFunction_new(zState_t st, const char *name,
                      void (*f)(zMemory_t, zMemory_t, zMemory_t, zMemory_t,
                                zMemory_t, zMemory_t, zMemory_t)) {
  zMapGroupFunction_t mpf = nNew(struct st_zMapGroupFunction_t);
  zMapGroupFunction_setName(mpf, zString_duplicate(name));
  zMapGroupFunction_setFunction6(mpf, f);
  return mpf;
}

zMapGroupFunction_t
zMapGroupFunction_new(zState_t st, const char *name,
                      void (*f)(zMemory_t, zMemory_t, zMemory_t, zMemory_t,
                                zMemory_t, zMemory_t, zMemory_t, zMemory_t)) {
  zMapGroupFunction_t mpf = nNew(struct st_zMapGroupFunction_t);
  zMapGroupFunction_setName(mpf, zString_duplicate(name));
  zMapGroupFunction_setFunction7(mpf, f);
  return mpf;
}

zMapFunction_t zMapFunction_new(zState_t st, const char *name,
                                void (*f)(zMemory_t, zMemory_t)) {
  zMapFunction_t mpf = nNew(struct st_zMapFunction_t);
  zMapFunction_setName(mpf, zString_duplicate(name));
  zMapFunction_setFunction1(mpf, f);
  return mpf;
}

zMapFunction_t zMapFunction_new(zState_t st, const char *name,
                                void (*f)(zMemory_t, zMemory_t, zMemory_t)) {
  zMapFunction_t mpf = nNew(struct st_zMapFunction_t);
  zMapFunction_setName(mpf, zString_duplicate(name));
  zMapFunction_setFunction2(mpf, f);
  return mpf;
}

zMapFunction_t zMapFunction_new(zState_t st, const char *name,
                                void (*f)(zMemory_t, zMemory_t, zMemory_t,
                                          zMemory_t)) {
  zMapFunction_t mpf = nNew(struct st_zMapFunction_t);
  zMapFunction_setName(mpf, zString_duplicate(name));
  zMapFunction_setFunction3(mpf, f);
  return mpf;
}

zMapFunction_t zMapFunction_new(zState_t st, const char *name,
                                void (*f)(zMemory_t, zMemory_t, zMemory_t,
                                          zMemory_t, zMemory_t)) {
  zMapFunction_t mpf = nNew(struct st_zMapFunction_t);
  zMapFunction_setName(mpf, zString_duplicate(name));
  zMapFunction_setFunction4(mpf, f);
  return mpf;
}

zMapFunction_t zMapFunction_new(zState_t st, const char *name,
                                void (*f)(zMemory_t, zMemory_t, zMemory_t,
                                          zMemory_t, zMemory_t, zMemory_t)) {
  zMapFunction_t mpf = nNew(struct st_zMapFunction_t);
  zMapFunction_setName(mpf, zString_duplicate(name));
  zMapFunction_setFunction5(mpf, f);
  return mpf;
}

zMapFunction_t zMapFunction_new(zState_t st, const char *name,
                                void (*f)(zMemory_t, zMemory_t, zMemory_t,
                                          zMemory_t, zMemory_t, zMemory_t,
                                          zMemory_t)) {
  zMapFunction_t mpf = nNew(struct st_zMapFunction_t);
  zMapFunction_setName(mpf, zString_duplicate(name));
  zMapFunction_setFunction6(mpf, f);
  return mpf;
}

zMapFunction_t zMapFunction_new(zState_t st, const char *name,
                                void (*f)(zMemory_t, zMemory_t, zMemory_t,
                                          zMemory_t, zMemory_t, zMemory_t,
                                          zMemory_t, zMemory_t)) {
  zMapFunction_t mpf = nNew(struct st_zMapFunction_t);
  zMapFunction_setName(mpf, zString_duplicate(name));
  zMapFunction_setFunction7(mpf, f);
  return mpf;
}

static void waitFor(zState_t st, zMemoryGroup_t mem) {
  // cudaStreamSynchronize(zState_getCopyToDeviceStream(st,
  // zMemoryGroup_getId(mem)));
  while (zMemoryGroup_getDeviceMemoryStatus(mem) < zMemoryStatus_cleanDevice) {
    continue;
  }
}

static void waitFor(zState_t st, zMemory_t mem) {
  while (zMemory_getDeviceMemoryStatus(mem) < zMemoryStatus_cleanDevice) {
    continue;
  }
}

void zMap(zState_t st, zMapGroupFunction_t gmapFun, zMemoryGroup_t gout,
          zMemoryGroup_t gin) {
  int nMems = zMemoryGroup_getMemoryCount(gout);

  tbb::parallel_for(0, nMems, [=](int ii) {
    zMemory_t out = zMemoryGroup_getMemory(gout, ii);
    zMemory_t in = zMemoryGroup_getMemory(gin, ii);

    zMapFunction_t memMapFun = zMapFunction_new(
        st, zMapFunction_getName(gmapFun), zMapFunction_getFunction1(gmapFun));
    zMap(st, memMapFun, out, in);
  });
  return;
}

void zMap(zState_t st, zMapGroupFunction_t gmapFun, zMemoryGroup_t gout,
          zMemoryGroup_t gin0, zMemoryGroup_t gin1) {
  int nMems = zMemoryGroup_getMemoryCount(gout);

  tbb::parallel_for(0, nMems, [=](int ii) {
    zMemory_t out = zMemoryGroup_getMemory(gout, ii);
    zMemory_t in0 = zMemoryGroup_getMemory(gin0, ii);
    zMemory_t in1 = zMemoryGroup_getMemory(gin1, ii);

    zMapFunction_t memMapFun = zMapFunction_new(
        st, zMapFunction_getName(gmapFun), zMapFunction_getFunction2(gmapFun));
    zMap(st, memMapFun, out, in0, in1);
  });
  return;
}

void zMap(zState_t st, zMapGroupFunction_t gmapFun, zMemoryGroup_t gout,
          zMemoryGroup_t gin0, zMemoryGroup_t gin1, zMemoryGroup_t gin2) {
  int nMems = zMemoryGroup_getMemoryCount(gout);

  tbb::parallel_for(0, nMems, [=](int ii) {
    zMemory_t out = zMemoryGroup_getMemory(gout, ii);
    zMemory_t in0 = zMemoryGroup_getMemory(gin0, ii);
    zMemory_t in1 = zMemoryGroup_getMemory(gin1, ii);
    zMemory_t in2 = zMemoryGroup_getMemory(gin2, ii);

    zMapFunction_t memMapFun = zMapFunction_new(
        st, zMapFunction_getName(gmapFun), zMapFunction_getFunction3(gmapFun));
    zMap(st, memMapFun, out, in0, in1, in2);
  });
  return;
}

void zMap(zState_t st, zMapGroupFunction_t gmapFun, zMemoryGroup_t gout,
          zMemoryGroup_t gin0, zMemoryGroup_t gin1, zMemoryGroup_t gin2,
          zMemoryGroup_t gin3) {
  int nMems = zMemoryGroup_getMemoryCount(gout);

  tbb::parallel_for(0, nMems, [=](int ii) {
    zMemory_t out = zMemoryGroup_getMemory(gout, ii);
    zMemory_t in0 = zMemoryGroup_getMemory(gin0, ii);
    zMemory_t in1 = zMemoryGroup_getMemory(gin1, ii);
    zMemory_t in2 = zMemoryGroup_getMemory(gin2, ii);
    zMemory_t in3 = zMemoryGroup_getMemory(gin3, ii);

    zMapFunction_t memMapFun = zMapFunction_new(
        st, zMapFunction_getName(gmapFun), zMapFunction_getFunction4(gmapFun));
    zMap(st, memMapFun, out, in0, in1, in2, in3);
  });
  return;
}

void zMap(zState_t st, zMapGroupFunction_t gmapFun, zMemoryGroup_t gout,
          zMemoryGroup_t gin0, zMemoryGroup_t gin1, zMemoryGroup_t gin2,
          zMemoryGroup_t gin3, zMemoryGroup_t gin4) {
  int nMems = zMemoryGroup_getMemoryCount(gout);

  tbb::parallel_for(0, nMems, [=](int ii) {
    zMemory_t out = zMemoryGroup_getMemory(gout, ii);
    zMemory_t in0 = zMemoryGroup_getMemory(gin0, ii);
    zMemory_t in1 = zMemoryGroup_getMemory(gin1, ii);
    zMemory_t in2 = zMemoryGroup_getMemory(gin2, ii);
    zMemory_t in3 = zMemoryGroup_getMemory(gin3, ii);
    zMemory_t in4 = zMemoryGroup_getMemory(gin4, ii);

    zMapFunction_t memMapFun = zMapFunction_new(
        st, zMapFunction_getName(gmapFun), zMapFunction_getFunction5(gmapFun));
    zMap(st, memMapFun, out, in0, in1, in2, in3, in4);
  });
  return;
}

void zMap(zState_t st, zMapGroupFunction_t gmapFun, zMemoryGroup_t gout,
          zMemoryGroup_t gin0, zMemoryGroup_t gin1, zMemoryGroup_t gin2,
          zMemoryGroup_t gin3, zMemoryGroup_t gin4, zMemoryGroup_t gin5) {
  int nMems = zMemoryGroup_getMemoryCount(gout);

  tbb::parallel_for(0, nMems, [=](int ii) {
    zMemory_t out = zMemoryGroup_getMemory(gout, ii);
    zMemory_t in0 = zMemoryGroup_getMemory(gin0, ii);
    zMemory_t in1 = zMemoryGroup_getMemory(gin1, ii);
    zMemory_t in2 = zMemoryGroup_getMemory(gin2, ii);
    zMemory_t in3 = zMemoryGroup_getMemory(gin3, ii);
    zMemory_t in4 = zMemoryGroup_getMemory(gin4, ii);
    zMemory_t in5 = zMemoryGroup_getMemory(gin5, ii);

    zMapFunction_t memMapFun = zMapFunction_new(
        st, zMapFunction_getName(gmapFun), zMapFunction_getFunction6(gmapFun));
    zMap(st, memMapFun, out, in0, in1, in2, in3, in4, in5);
  });
  return;
}

void zMap(zState_t st, zMapGroupFunction_t gmapFun, zMemoryGroup_t gout,
          zMemoryGroup_t gin0, zMemoryGroup_t gin1, zMemoryGroup_t gin2,
          zMemoryGroup_t gin3, zMemoryGroup_t gin4, zMemoryGroup_t gin5,
          zMemoryGroup_t gin6) {
  int nMems = zMemoryGroup_getMemoryCount(gout);

  tbb::parallel_for(0, nMems, [=](int ii) {
    zMemory_t out = zMemoryGroup_getMemory(gout, ii);
    zMemory_t in0 = zMemoryGroup_getMemory(gin0, ii);
    zMemory_t in1 = zMemoryGroup_getMemory(gin1, ii);
    zMemory_t in2 = zMemoryGroup_getMemory(gin2, ii);
    zMemory_t in3 = zMemoryGroup_getMemory(gin3, ii);
    zMemory_t in4 = zMemoryGroup_getMemory(gin4, ii);
    zMemory_t in5 = zMemoryGroup_getMemory(gin5, ii);
    zMemory_t in6 = zMemoryGroup_getMemory(gin6, ii);

    zMapFunction_t memMapFun = zMapFunction_new(
        st, zMapFunction_getName(gmapFun), zMapFunction_getFunction7(gmapFun));
    zMap(st, memMapFun, out, in0, in1, in2, in3, in4, in5, in6);
  });
  return;
}

void zMap(zState_t st, zMapFunction_t mapFun, zMemory_t out, zMemory_t in) {

  zCUDA_copyToDevice(in);

  while (!zMemory_deviceMemoryAllocatedQ(out)) {
    continue;
  }

  waitFor(st, in);

  zMemory_setDeviceMemoryStatus(out, zMemoryStatus_dirtyDevice);
  zMapFunction_getFunction1(mapFun)(out, in);

  return;
}

void zMap(zState_t st, zMapFunction_t mapFun, zMemory_t out, zMemory_t in0,
          zMemory_t in1) {

  zCUDA_copyToDevice(in0);
  zCUDA_copyToDevice(in1);

  while (!zMemory_deviceMemoryAllocatedQ(out)) {
    continue;
  }

  waitFor(st, in0);
  waitFor(st, in1);

  zMemory_setDeviceMemoryStatus(out, zMemoryStatus_dirtyDevice);
  zMapFunction_getFunction2(mapFun)(out, in0, in1);

  return;
}

void zMap(zState_t st, zMapFunction_t mapFun, zMemory_t out, zMemory_t in0,
          zMemory_t in1, zMemory_t in2) {

  zCUDA_copyToDevice(in0);
  zCUDA_copyToDevice(in1);
  zCUDA_copyToDevice(in2);

  while (!zMemory_deviceMemoryAllocatedQ(out)) {
    continue;
  }

  waitFor(st, in0);
  waitFor(st, in1);
  waitFor(st, in2);

  zMemory_setDeviceMemoryStatus(out, zMemoryStatus_dirtyDevice);
  zMapFunction_getFunction3(mapFun)(out, in0, in1, in2);

  return;
}

void zMap(zState_t st, zMapFunction_t mapFun, zMemory_t out, zMemory_t in0,
          zMemory_t in1, zMemory_t in2, zMemory_t in3) {

  zCUDA_copyToDevice(in0);
  zCUDA_copyToDevice(in1);
  zCUDA_copyToDevice(in2);
  zCUDA_copyToDevice(in3);

  while (!zMemory_deviceMemoryAllocatedQ(out)) {
    continue;
  }

  waitFor(st, in0);
  waitFor(st, in1);
  waitFor(st, in2);
  waitFor(st, in3);

  zMemory_setDeviceMemoryStatus(out, zMemoryStatus_dirtyDevice);
  zMapFunction_getFunction4(mapFun)(out, in0, in1, in2, in3);

  return;
}

void zMap(zState_t st, zMapFunction_t mapFun, zMemory_t out, zMemory_t in0,
          zMemory_t in1, zMemory_t in2, zMemory_t in3, zMemory_t in4) {

  zCUDA_copyToDevice(in0);
  zCUDA_copyToDevice(in1);
  zCUDA_copyToDevice(in2);
  zCUDA_copyToDevice(in3);
  zCUDA_copyToDevice(in4);

  while (!zMemory_deviceMemoryAllocatedQ(out)) {
    continue;
  }

  waitFor(st, in0);
  waitFor(st, in1);
  waitFor(st, in2);
  waitFor(st, in3);
  waitFor(st, in4);

  zMemory_setDeviceMemoryStatus(out, zMemoryStatus_dirtyDevice);
  zMapFunction_getFunction5(mapFun)(out, in0, in1, in2, in3, in4);

  return;
}

void zMap(zState_t st, zMapFunction_t mapFun, zMemory_t out, zMemory_t in0,
          zMemory_t in1, zMemory_t in2, zMemory_t in3, zMemory_t in4,
          zMemory_t in5) {

  zCUDA_copyToDevice(in0);
  zCUDA_copyToDevice(in1);
  zCUDA_copyToDevice(in2);
  zCUDA_copyToDevice(in3);
  zCUDA_copyToDevice(in4);
  zCUDA_copyToDevice(in5);

  while (!zMemory_deviceMemoryAllocatedQ(out)) {
    continue;
  }

  waitFor(st, in0);
  waitFor(st, in1);
  waitFor(st, in2);
  waitFor(st, in3);
  waitFor(st, in4);
  waitFor(st, in5);

  zMemory_setDeviceMemoryStatus(out, zMemoryStatus_dirtyDevice);
  zMapFunction_getFunction6(mapFun)(out, in0, in1, in2, in3, in4, in5);

  return;
}

void zMap(zState_t st, zMapFunction_t mapFun, zMemory_t out, zMemory_t in0,
          zMemory_t in1, zMemory_t in2, zMemory_t in3, zMemory_t in4,
          zMemory_t in5, zMemory_t in6) {

  zCUDA_copyToDevice(in0);
  zCUDA_copyToDevice(in1);
  zCUDA_copyToDevice(in2);
  zCUDA_copyToDevice(in3);
  zCUDA_copyToDevice(in4);
  zCUDA_copyToDevice(in5);
  zCUDA_copyToDevice(in6);

  while (!zMemory_deviceMemoryAllocatedQ(out)) {
    continue;
  }

  waitFor(st, in0);
  waitFor(st, in1);
  waitFor(st, in2);
  waitFor(st, in3);
  waitFor(st, in4);
  waitFor(st, in5);
  waitFor(st, in6);

  zMemory_setDeviceMemoryStatus(out, zMemoryStatus_dirtyDevice);
  zMapFunction_getFunction7(mapFun)(out, in0, in1, in2, in3, in4, in5, in6);

  return;
}
