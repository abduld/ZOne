
#ifndef __ZFUNCTION_H__
#define __ZFUNCTION_H__

struct st_zFunction_t {
  string name;
  size_t nInstructions;
  size_t nCycles;
  zMapFunction_t mf;
  zMapGroupFunction_t mgf;
};

#define zFunction_getName(fun) ((fun)->name)
#define zFunction_getNunInstructions(fun) ((fun)->nInstructions)
#define zFunction_getNumCycles(fun) ((fun)->nCycles)

#define zFunction_setName(fun, val)                                 \
  (zFunction_getName(fun) = val)
#define zFunction_setNunInstructions(fun, val)                      \
  (zFunction_getNunInstructions(fun) = val)
#define zFunction_setNumCycles(fun, val)                            \
  (zFunction_getNumCycles(fun) = val)


zFunction_t zFunction_new(const char * name, zMapFunction_t mf, zMapGroupFunction_t mgf);

#endif /* __ZFUNCTION_H__ */
