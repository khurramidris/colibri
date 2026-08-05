#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../replay_io.h"

#define CHECK(x) do { if(!(x)){ \
    fprintf(stderr,"CHECK failed at %s:%d: %s\n",__FILE__,__LINE__,#x); return 1; \
} } while(0)

int main(void){
    ReplayIoSnapshot s={0};
    const char *valid=
        "rchar: 12345\n"
        "wchar: 99\n"
        "syscr: 17\n"
        "syscw: 2\n"
        "read_bytes: 8192\n"
        "write_bytes: 0\n"
        "cancelled_write_bytes: 0\n";
    CHECK(replay_io_parse_text(valid,&s));
    CHECK(s.available==1);
    CHECK(s.rchar==12345);
    CHECK(s.read_bytes==8192);
    CHECK(s.read_syscalls==17);

    const char *unordered=
        "read_bytes:    4096\n"
        "syscr: 3\n"
        "ignored: 42\n"
        "rchar: 700\n";
    CHECK(replay_io_parse_text(unordered,&s));
    CHECK(s.rchar==700 && s.read_bytes==4096 && s.read_syscalls==3);

    CHECK(!replay_io_parse_text("rchar: 1\nsyscr: 2\n",&s));
    CHECK(!replay_io_parse_text("rchar: nope\nread_bytes: 1\nsyscr: 2\n",&s));
    CHECK(!replay_io_parse_text("rchar: 1x\nread_bytes: 1\nsyscr: 2\n",&s));
    CHECK(!replay_io_parse_text("rchar: -1\nread_bytes: 1\nsyscr: 2\n",&s));
    CHECK(!replay_io_parse_text(
        "rchar: 18446744073709551616\nread_bytes: 1\nsyscr: 2\n",&s));
    CHECK(!replay_io_parse_text(
        "rchar: 1\nrchar: 2\nread_bytes: 1\nsyscr: 2\n",&s));

    ReplayIoSnapshot a={100,4096,10,1};
    ReplayIoSnapshot b={160,12288,14,1};
    ReplayIoDelta d={0};
    CHECK(replay_io_delta(&a,&b,&d));
    CHECK(d.available==1);
    CHECK(d.rchar==60 && d.read_bytes==8192 && d.read_syscalls==4);

    b.read_bytes=1024;
    CHECK(!replay_io_delta(&a,&b,&d));
    b.read_bytes=12288; b.available=0;
    CHECK(!replay_io_delta(&a,&b,&d));

    ReplayIoSnapshot live={0};
#if defined(__linux__)
    CHECK(replay_io_snapshot(&live));
    CHECK(live.available==1);
#else
    CHECK(!replay_io_snapshot(&live));
    CHECK(live.available==0);
#endif

    puts("test_replay_io: ok");
    return 0;
}
