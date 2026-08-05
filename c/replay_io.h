#ifndef COLIBRI_REPLAY_IO_H
#define COLIBRI_REPLAY_IO_H

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct {
    uint64_t rchar;
    uint64_t read_bytes;
    uint64_t read_syscalls;
    int available;
} ReplayIoSnapshot;

typedef struct {
    uint64_t rchar;
    uint64_t read_bytes;
    uint64_t read_syscalls;
    int available;
} ReplayIoDelta;

static int replay_io_parse_u64_field(const char *text, const char *name, uint64_t *out){
    if(!text || !name || !*name || !out) return 0;
    size_t name_n=strlen(name);
    const char *line=text;
    int found=0;
    while(*line){
        const char *end=strchr(line,'\n');
        if(!end) end=line+strlen(line);
        if((size_t)(end-line)>name_n && memcmp(line,name,name_n)==0 && line[name_n]==':'){
            if(found) return 0; /* duplicate fields are ambiguous */
            const char *value=line+name_n+1;
            while(value<end && isspace((unsigned char)*value)) value++;
            if(value==end || *value=='-') return 0;
            errno=0;
            char *tail=NULL;
            unsigned long long parsed=strtoull(value,&tail,10);
            if(errno==ERANGE || tail==value || tail>end) return 0;
            while(tail<end && isspace((unsigned char)*tail)) tail++;
            if(tail!=end) return 0;
            *out=(uint64_t)parsed;
            found=1;
        }
        if(!*end) break;
        line=end+1;
    }
    return found;
}

static int replay_io_parse_text(const char *text, ReplayIoSnapshot *out){
    if(!out) return 0;
    memset(out,0,sizeof(*out));
    if(!replay_io_parse_u64_field(text,"rchar",&out->rchar)) return 0;
    if(!replay_io_parse_u64_field(text,"read_bytes",&out->read_bytes)) return 0;
    if(!replay_io_parse_u64_field(text,"syscr",&out->read_syscalls)) return 0;
    out->available=1;
    return 1;
}

static int replay_io_snapshot(ReplayIoSnapshot *out){
    if(!out) return 0;
    memset(out,0,sizeof(*out));
#if defined(__linux__)
    FILE *f=fopen("/proc/self/io","rb");
    if(!f) return 0;
    char buf[4096];
    size_t n=fread(buf,1,sizeof(buf)-1,f);
    int bad=ferror(f);
    fclose(f);
    if(bad || n==0 || n>=sizeof(buf)) return 0;
    buf[n]='\0';
    return replay_io_parse_text(buf,out);
#else
    return 0;
#endif
}

static int replay_io_delta(const ReplayIoSnapshot *before,
                           const ReplayIoSnapshot *after,
                           ReplayIoDelta *out){
    if(!out) return 0;
    memset(out,0,sizeof(*out));
    if(!before || !after || !before->available || !after->available) return 0;
    if(after->rchar<before->rchar || after->read_bytes<before->read_bytes ||
       after->read_syscalls<before->read_syscalls) return 0;
    out->rchar=after->rchar-before->rchar;
    out->read_bytes=after->read_bytes-before->read_bytes;
    out->read_syscalls=after->read_syscalls-before->read_syscalls;
    out->available=1;
    return 1;
}

#endif
