#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef int (*rwlock_fn_t)(pthread_rwlock_t *);

static rwlock_fn_t real_rdlock;
static rwlock_fn_t real_wrlock;
static int log_fd = -1;
static uint64_t min_wait_ns = 1000000ULL;
static _Thread_local int in_probe;

static uint64_t raw_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void init_probe(void) {
    static atomic_int initialized = 0;
    int expected = 0;
    if (!atomic_compare_exchange_strong(&initialized, &expected, 1)) {
        while (atomic_load(&initialized) == 1) {
        }
        return;
    }

    real_rdlock = (rwlock_fn_t)dlsym(RTLD_NEXT, "pthread_rwlock_rdlock");
    real_wrlock = (rwlock_fn_t)dlsym(RTLD_NEXT, "pthread_rwlock_wrlock");

    const char *min_us = getenv("RWLOCK_PROBE_MIN_US");
    if (min_us && min_us[0]) {
        char *end = NULL;
        unsigned long long value = strtoull(min_us, &end, 10);
        if (end != min_us && value > 0) {
            min_wait_ns = value * 1000ULL;
        }
    }

    const char *path = getenv("RWLOCK_PROBE_LOG");
    if (path && path[0]) {
        log_fd = open(path, O_CREAT | O_WRONLY | O_APPEND | O_CLOEXEC, 0644);
    }

    atomic_store(&initialized, 2);
}

static void log_wait(const char *op, pthread_rwlock_t *lock, uint64_t start_ns,
                     uint64_t end_ns, int ret, void *caller) {
    if (log_fd < 0 || end_ns <= start_ns || end_ns - start_ns < min_wait_ns) {
        return;
    }

    Dl_info info;
    const char *object = "?";
    const char *symbol = "?";
    uintptr_t offset = 0;
    if (dladdr(caller, &info) != 0) {
        if (info.dli_fname) {
            object = info.dli_fname;
        }
        if (info.dli_sname) {
            symbol = info.dli_sname;
            offset = (uintptr_t)caller - (uintptr_t)info.dli_saddr;
        } else if (info.dli_fbase) {
            offset = (uintptr_t)caller - (uintptr_t)info.dli_fbase;
        }
    }

    char buf[1024];
    int len = snprintf(buf, sizeof(buf),
                       "%llu,tid=%ld,op=%s,wait_us=%llu,ret=%d,lock=%p,caller=%p,obj=%s,sym=%s,off=0x%lx\n",
                       (unsigned long long)end_ns, (long)syscall(SYS_gettid),
                       op, (unsigned long long)((end_ns - start_ns) / 1000ULL),
                       ret, (void *)lock, caller, object, symbol,
                       (unsigned long)offset);
    if (len > 0) {
        if (len > (int)sizeof(buf)) {
            len = (int)sizeof(buf);
        }
        (void)write(log_fd, buf, (size_t)len);
    }
}

int pthread_rwlock_rdlock(pthread_rwlock_t *rwlock) {
    if (!real_rdlock) {
        init_probe();
    }
    if (in_probe || !real_rdlock) {
        errno = ENOSYS;
        return ENOSYS;
    }
    in_probe = 1;
    uint64_t start_ns = raw_time_ns();
    int ret = real_rdlock(rwlock);
    uint64_t end_ns = raw_time_ns();
    void *caller = __builtin_return_address(0);
    log_wait("rdlock", rwlock, start_ns, end_ns, ret, caller);
    in_probe = 0;
    return ret;
}

int pthread_rwlock_wrlock(pthread_rwlock_t *rwlock) {
    if (!real_wrlock) {
        init_probe();
    }
    if (in_probe || !real_wrlock) {
        errno = ENOSYS;
        return ENOSYS;
    }
    in_probe = 1;
    uint64_t start_ns = raw_time_ns();
    int ret = real_wrlock(rwlock);
    uint64_t end_ns = raw_time_ns();
    void *caller = __builtin_return_address(0);
    log_wait("wrlock", rwlock, start_ns, end_ns, ret, caller);
    in_probe = 0;
    return ret;
}
