

#ifndef zError_define
#error "zError_define is not defined
#endif /* zError_define */

zError_define(success, "success")
zError_define(uv, "uv_error")
zError_define(fail, "failure")
zError_define(unknown, "unknown")
zError_define(memoryAllocation, "memoryAllocation")

#undef zError_define
