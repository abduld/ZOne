

#ifndef __ZREAD_H__
#define __ZREAD_H__

/* We are going to assume that the format of the files is that
the first line will contain the dimensions and the second
line contains the data comma sperated */

zMemoryGroup_t zReadBit8Array(zState_t st, const char *file, int rank,
                              size_t *dims);
zMemoryGroup_t zReadInt32Array(zState_t st, const char *file, int rank,
                               size_t *dims);
zMemoryGroup_t zReadInt64Array(zState_t st, const char *file, int rank,
                               size_t *dims);
zMemoryGroup_t zReadFloatArray(zState_t st, const char *file, int rank,
                               size_t *dims);
zMemoryGroup_t zReadDoubleArray(zState_t st, const char *file, int rank,
                                size_t *dims);

#endif /* __ZREAD_H__ */
