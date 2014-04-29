
#ifndef __ZOPTIONS_H__
#define __ZOPTIONS_H__

#define zTrue true
#define zFalse false

#define Z_CONFIG_DEBUG True
#define Z_CONFIG_EAGER_COPY True
//#define Z_CONFIG_MAX_CHUNKS 4
#define Z_CONFIG_SYNC_CUDA_TIME 0

typedef bool zBool_t;
typedef int zInteger_t;
typedef float zReal_t;

#ifdef _MSC_VER
#define __func__ __FUNCTION__
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS 1
#endif /* _CRT_SECURE_NO_WARNINGS */
#define _CRT_SECURE_NO_DEPRECATE 1
#define _CRT_NONSTDC_NO_DEPRECATE 1
#endif /* _MSC_VER */

#endif /* __ZOPTIONS_H__ */
