

#include "z.h"

void getCPUInfo(zState_t st) {
  uv_cpu_info(&zState_getCPUInfo(st), &zState_getCPUCount(st));
}

void freeCPUInfo(zState_t st) {
  uv_free_cpu_info(zState_getCPUInfo(st), zState_getCPUCount(st));
}


void zState_addMemoryGroup(zState_t st, zMemoryGroup_t mg) {
	return ;
}