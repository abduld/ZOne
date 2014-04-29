
#ifndef __ZFUNCTION_H__
#define __ZFUNCTION_H__

struct st_zFunctionInformation_t {
  string name;
  size_t nInstructions;
  size_t nCycles;
};

#define zFunctionInformation_getName(fun) ((fun)->name)
#define zFunctionInformation_getNunInstructions(fun) ((fun)->nInstructions)
#define zFunctionInformation_getNumCycles(fun) ((fun)->nCycles)

#define zFunctionInformation_setName(fun, val)                                 \
  (zFunctionInformation_getName(fun) = val)
#define zFunctionInformation_setNunInstructions(fun, val)                      \
  (zFunctionInformation_getNunInstructions(fun) = val)
#define zFunctionInformation_setNumCycles(fun, val)                            \
  (zFunctionInformation_getNumCycles(fun) = val)

#endif /* __ZFUNCTION_H__ */
