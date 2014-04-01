#include    <z.h>
 
zFile_t zFile_new(const char * path, int flags) {
    zFile_t file;
     
    if (path == NULL) {
        return NULL;
    }
 
    file = zNew(struct st_zFile_t);
 
    zFile_setPath(file, zString_duplicate(path));
    zFile_setState(file, NULL);
    zFile_setFileHandle(file, -1);
    zFile_setFlags(file, flags);
    zFile_setOpenedQ(file, zFalse);
 
    zFile_setOffset(file, 0);
 
    return file;
}
 
void zFile_delete(zFile_t file) {
    if (file != NULL) {
        if (zFile_getPath(file)) {
            zDelete(zFile_getPath(file));
        }
        if (zFile_getFileHandle(file) != -1) {
            zFile_close(file);
        }
        zDelete(file);
    }
    return ;
}
 
void zFile_open(zFile_t file) {
    zState_t ctx;
    uv_loop_t * loop;
     
    if (zFile_getOpenedQ(file) == zTrue) {
        return ;
    }
     
    ctx = zFile_getState(file);
    zAssert(ctx != NULL);
 
    loop = zState_getLoop(ctx);
 
    if (zFile_existsQ(zFile_getPath(file))) {
        uv_fs_t req;
        uv_fs_unlink(loop, &req, zFile_getPath(file), NULL);
        uv_fs_req_cleanup(&req);
    }
 
    uv_fs_open(loop, &zFile_getOpenRequest(file), zFile_getPath(file), zFile_getFlags(file),  S_IREAD | S_IWRITE, NULL);
     
    uv_fs_req_cleanup(&zFile_getOpenRequest(file));
     
    zFile_setFileHandle(file, (uv_file) zFile_getOpenRequest(file).result);
    zFile_setOpenedQ(file, zTrue);
 
    zState_mutexed(ctx, {
        zLog(zState_getLogger(ctx), ERROR, "Opening file.");
    });
 
    return ;
}

size_t zFile_readChunk(zFile_t file, void * chunk, size_t sz, uv_fs_cb cb) {
    // XXX TODO
}

void zFile_close(zFile_t file) {
    zState_t ctx;
    uv_loop_t * loop;
    uv_fs_t closeRequest;
 
    if (zFile_getOpenedQ(file) == zFalse) {
        return ;
    }
     
    ctx = zFile_getState(file);
    zAssert(ctx != NULL);
 
    loop = zState_getLoop(ctx);
 
    uv_fs_close(loop, &closeRequest, zFile_getFileHandle(file), NULL);
     
    zState_mutexed(ctx, {
        zLog(zState_getLogger(ctx), ERROR, "Closing file.");
    });
 
    zFile_setFileHandle(file, -1);
 
    return ;
}
 
void zFile_write(zFile_t file, const char * text) {
    zState_t ctx;
    uv_loop_t * loop;
    size_t textLength;
     
    if (zFile_getOpenedQ(file) == zFalse) {
        zFile_open(file);
    }
 
    if (!zFile_existsQ(zFile_getPath(file))) {
        return ;
    }
 
    ctx = zFile_getState(file);
    zAssert(ctx != NULL);
 
    loop = zState_getLoop(ctx);
     
    uv_fs_req_cleanup(&zFile_getOpenRequest(file));
 
    textLength = strlen(text);
 
    uv_fs_write(loop, &zFile_getWriteRequest(file), zFile_getFileHandle(file), (char *) text, textLength, zFile_getOffset(file), NULL);
     
    zState_mutexed(ctx, {
        zLog(zState_getLogger(ctx), ERROR, "Writing to file.");
    });
     
    zFile_getOffset(file) += textLength;
 
    uv_fs_req_cleanup(&zFile_getWriteRequest(file));
 
    return ;
}
