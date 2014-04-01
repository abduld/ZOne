
#include	<z.h>

struct st_zStringBuffer_t {
	int id;
	char * buf;
	size_t curLen;
	size_t maxLen;
};

#define zStringBuffer_initialLength					64
#define zStringBuffer_expandMultiplier					2

#define zStringBuffer_getId(sb)						((sb)->id)
#define zStringBuffer_getCurrentLength(sb)				((sb)->curLen)
#define zStringBuffer_getBuffer(sb)					((sb)->buf)
#define zStringBuffer_getBufferChar(sb, ii)			(zStringBuffer_getBuffer(sb)[ii])
#define zStringBuffer_getMaxLength(sb)					((sb)->maxLen)

#define zStringBuffer_setCurrentLength(sb, val)		(zStringBuffer_getCurrentLength(sb) = val)
#define zStringBuffer_setBuffer(sb, val)				(zStringBuffer_getBuffer(sb) = val)
#define zStringBuffer_setBufferChar(sb, ii, val)		(zStringBuffer_getBufferChar(sb, ii) = val)
#define zStringBuffer_setMaxLength(sb, val)			(zStringBuffer_getMaxLength(sb) = val)


zStringBuffer_t zStringBuffer_new(void) {
	char * buf;
	size_t size = zStringBuffer_initialLength;
	zStringBuffer_t sb = zNew(struct st_zStringBuffer_t);
	zStringBuffer_setId(sb, -1);
	zStringBuffer_setCurrentLength(sb, 0);
	zStringBuffer_setMaxLength(sb, size);
	buf = zNewArray(char, zStringBuffer_initialLength);
	assert(buf != NULL);
	zStringBuffer_setBuffer(sb, buf);
	memset(buf, '\0', size);
	return sb;
}

zStringBuffer_t zStringBuffer_initialize(size_t sz) {
	zStringBuffer_t sb = zNew(struct st_zStringBuffer_t);
	zStringBuffer_setId(sb, -1);
	zStringBuffer_setCurrentLength(sb, 0);
	zStringBuffer_setMaxLength(sb, sz);
	zStringBuffer_setBuffer(sb, zReallocArray(char, NULL, sz));
	return sb;
}

void zStringBuffer_deleteStructure(zStringBuffer_t sb) {
  if (sb) {
    zDelete(sb);
  }
  return ;
}

void zStringBuffer_delete(zStringBuffer_t sb) {
	if (sb) {
		if (zStringBuffer_getBuffer(sb)) {
			zDelete(zStringBuffer_getBuffer(sb));
		}
		zDelete(sb);
	}
	return ;
}

static zBool zStringBuffer_expand(zStringBuffer_t sb, size_t len) {
	if (sb) {
		if (zStringBuffer_getCurrentLength(sb) + len < zStringBuffer_getMaxLength(sb) - 1) {
			return zTrue;
		} else {
			size_t oldSize = zStringBuffer_getMaxLength(sb);
			size_t newSize = zStringBuffer_expandMultiplier * (oldSize + 1);
			char * newMemory;

			while (newSize < len) {
				newSize *= zStringBuffer_expandMultiplier;
			}
			
			newMemory = zReallocArray(char, zStringBuffer_getBuffer(sb), newSize);
			if (newMemory == NULL) {
				/* out of memory */
				return zFalse;
			} else {
				size_t ii;
				char * buf = newMemory + oldSize;
				zStringBuffer_setBuffer(sb, newMemory);
				for (ii = oldSize; ii < newSize; ii++) {
					*buf++ = '\0';
				}
				zStringBuffer_setMaxLength(sb, newSize);
				return zTrue;
			}
		}
	} else {
		return zFalse;
	}
}

void zStringBuffer_setId(zStringBuffer_t sb, int id) {
	if (sb) {
		zStringBuffer_getId(sb) = id;
	}
	return ;
}

int zStringBuffer_id(zStringBuffer_t sb) {
	if (sb) {
		return zStringBuffer_getId(sb);
	} else {
		return -1;
	}
}

size_t zStringBuffer_length(zStringBuffer_t sb) {
	if (sb) {
		return zStringBuffer_getCurrentLength(sb);
	} else {
		return 0;
	}
}

void zStringBuffer_append(zStringBuffer_t sb, const char * msg) {
	if (sb) {
		size_t slen = strlen(msg);

		if (zStringBuffer_expand(sb, slen)) {
			size_t ii = 0;
			char * buf = zStringBuffer_getBuffer(sb) + zStringBuffer_getCurrentLength(sb);

			while (ii < slen) {
				*buf++ = *msg++;
				ii++;
			}
			*buf = '\0';

			zStringBuffer_setCurrentLength(sb, zStringBuffer_getCurrentLength(sb) + slen);

			assert(zStringBuffer_getCurrentLength(sb) < zStringBuffer_getMaxLength(sb));
		}
	}
	return ;
}

void zStringBuffer_join(zStringBuffer_t sb, zStringBuffer_t from) {
	if (sb) {
		size_t slen = zStringBuffer_length(from);

		if (zStringBuffer_expand(sb, slen)) {
			size_t ii = 0;
			char * msg = zStringBuffer_getBuffer(from);
			char * buf = zStringBuffer_getBuffer(sb) + zStringBuffer_getCurrentLength(sb);

			while (ii < slen) {
				*buf++ = *msg++;
				ii++;
			}
			*buf = '\0';

			zStringBuffer_setCurrentLength(sb, zStringBuffer_getCurrentLength(sb) + slen);

			assert(zStringBuffer_getCurrentLength(sb) < zStringBuffer_getMaxLength(sb));
		}
	}
	return ;
}

zStringBuffer_t zStringBuffer_reverse(zStringBuffer_t from) {
	if (from) {
		zStringBuffer_t to = zStringBuffer_initialize(zStringBuffer_getCurrentLength(from) + 1);
		char * toBuf, * fromBuf;
		size_t ii = 0, len;

		len = zStringBuffer_getCurrentLength(from);

		toBuf = zStringBuffer_getBuffer(to) + len - 1;
		fromBuf = zStringBuffer_getBuffer(from);

		while (ii < len) {
			*toBuf-- = *fromBuf++;
			ii--;
		}
		toBuf[len] = '\0';
	}
	return NULL;
}

zStringBuffer_t zStringBuffer_take(zStringBuffer_t from, size_t start, size_t end) {
	size_t ii;
	char * toBuf, * fromBuf;
	zStringBuffer_t to = NULL;	

	if (from == NULL ||
		start >= zStringBuffer_getCurrentLength(from) ||
		end >= zStringBuffer_getCurrentLength(from)) {
		return NULL;
	}

	if (end < 0) {
		end += zStringBuffer_getCurrentLength(from);
	}

	if (start < 0) {
		start += zStringBuffer_getCurrentLength(from);
	}

	if (end == start) {
		to = zStringBuffer_initialize(1);
		zStringBuffer_setBufferChar(to, 0, '\0');
	} else if (end > start) {
		to = zStringBuffer_initialize(end - start + 1);
		toBuf = zStringBuffer_getBuffer(to);
		fromBuf = zStringBuffer_getBuffer(from) + start;
		for (ii = start; ii < end; ii++) {
			*toBuf++ = *fromBuf++;
		}
	}

	return to;
}

char * zStringBuffer_toCString(zStringBuffer_t sb) {
	if (sb) {
		return zStringBuffer_getBuffer(sb);
	} else {
		return NULL;
	}
}




