

#ifndef __ZSTRING_BUFFER_H__
#define __ZSTRING_BUFFER_H__

zStringBuffer_t zStringBuffer_new(void);
zStringBuffer_t zStringBuffer_initialize(size_t sz);
void zStringBuffer_delete(zStringBuffer_t sb);
void zStringBuffer_deleteStructure(zStringBuffer_t sb);
static zBool_t zStringBuffer_expand(zStringBuffer_t sb, size_t len);
void zStringBuffer_setId(zStringBuffer_t sb, int id);
int zStringBuffer_id(zStringBuffer_t sb);
size_t zStringBuffer_length(zStringBuffer_t sb);
void zStringBuffer_append(zStringBuffer_t sb, const char *msg);
void zStringBuffer_join(zStringBuffer_t sb, zStringBuffer_t from);
zStringBuffer_t zStringBuffer_reverse(zStringBuffer_t from);
zStringBuffer_t zStringBuffer_take(zStringBuffer_t from, size_t start,
                                   size_t end);
char *zStringBuffer_toCString(zStringBuffer_t sb);

#endif /* __ZSTRING_BUFFER_H__ */
