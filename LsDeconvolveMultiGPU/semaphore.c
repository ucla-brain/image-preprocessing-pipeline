// semaphore.c - Cross-platform shared memory semaphore for MATLAB MEX
//
// Author: Keivan Moradi (2025) @ B.R.A.I.N. Lab, UCLA
// License: GPLv3
/*
 * This file implements a named semaphore mechanism that works on both
 * Windows and POSIX (Linux/macOS) platforms using shared memory and
 * synchronization primitives. It is designed for use in parallel and
 * multi-process MATLAB environments where native semaphores are either
 * unavailable or incompatible.
 *
 * Core Concepts:
 * - Each semaphore is identified by a unique integer key.
 * - A shared memory region holds the semaphore structure (count, max, flags).
 * - Mutual exclusion is achieved using a cross-platform mutex.
 * - Notification of waiting threads/processes is done using platform-specific events:
 *     - Windows: Event objects + Mutex + memory mapping
 *     - POSIX: pthread mutex + condition variable in shared memory
 *
 * Supported Operations:
 * 1. create: Initializes (or recreates) the semaphore. If it already exists,
 *    it is reset and any waiting threads are released. Safe to call multiple times.
 *
 * 2. post: Increments the semaphore count. If already at maximum, a warning is issued
 *    but the operation is still successful. Waiting threads are signaled.
 *
 * 3. wait: Decrements the semaphore count, blocking the caller if count is zero.
 *    Wakes up when post() occurs or if the semaphore is destroyed.
 *
 * 4. destroy: Marks the semaphore as terminated and releases all waiting threads.
 *    Also cleans up shared memory and synchronization primitives.
 *
 * Platform Compatibility:
 * - Windows implementation uses named file mappings, mutexes, and events.
 * - POSIX implementation uses shm_open, mmap, pthread_mutex, and pthread_cond.
 *
 * All behaviors are carefully matched across platforms to ensure identical
 * semantics and reliability under concurrent access.
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include "mex.h"

#define SEMAPHORE_KEY_NAME_FMT "MATLAB_MEX_SEM_%d"

#if defined(_WIN32)
#  define IS_WINDOWS 1
#  include <windows.h>
#  include <strsafe.h>
#  define MUTEX_NAME_PREFIX "Local\\"
#  define EVENT_NAME_PREFIX "Local\\"
#  define SHM_NAME_PREFIX   "Local\\"
#else
#  define IS_WINDOWS 0
#  include <pthread.h>
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  include <sys/file.h>
#  include <sys/types.h>
#  include <sys/time.h>
#endif

typedef struct {
    int initialized;
    int count;
    int max;
    volatile int terminate;
#if !IS_WINDOWS
    pthread_mutex_t mutex;
    pthread_cond_t cond;
#endif
} shared_semaphore_t;

#if IS_WINDOWS

typedef struct {
    shared_semaphore_t* sem;
    HANDLE hMap;
    HANDLE hMutex;
    HANDLE hEvent;
    BOOL is_new;
} semaphore_map_result_t;

static void close_handles(semaphore_map_result_t* mapped) {
    if (mapped->sem) UnmapViewOfFile(mapped->sem);
    if (mapped->hMutex) CloseHandle(mapped->hMutex);
    if (mapped->hEvent) CloseHandle(mapped->hEvent);
    if (mapped->hMap)   CloseHandle(mapped->hMap);
}

static semaphore_map_result_t map_shared_semaphore(int key, BOOL create) {
    semaphore_map_result_t result = {0};
    size_t size = sizeof(shared_semaphore_t);
    char basename[64], shmname[128], mname[128], ename[128];

    snprintf(basename, sizeof(basename), SEMAPHORE_KEY_NAME_FMT, key);
    StringCchPrintfA(shmname, 128, "%s%s", SHM_NAME_PREFIX, basename);
    StringCchPrintfA(mname, 128, "%s%s_MUTEX", MUTEX_NAME_PREFIX, basename);
    StringCchPrintfA(ename, 128, "%s%s_EVENT", EVENT_NAME_PREFIX, basename);

    if (create) {
        result.hMap = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, (DWORD)size, shmname);
        if (!result.hMap)
            mexErrMsgIdAndTxt("semaphore:create", "Failed to create file mapping");
        result.is_new = (GetLastError() != ERROR_ALREADY_EXISTS);
    } else {
        result.hMap = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, shmname);
        if (!result.hMap)
            mexErrMsgIdAndTxt("semaphore:notfound", "Semaphore %d not found. Use 'create' first.", key);
        result.is_new = FALSE;
    }

    void* base = MapViewOfFile(result.hMap, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!base)
        mexErrMsgIdAndTxt("semaphore:map", "Failed to map shared memory for key %d", key);

    result.sem = (shared_semaphore_t*)base;

    result.hMutex = CreateMutexA(NULL, FALSE, mname);
    if (!result.hMutex)
        mexErrMsgIdAndTxt("semaphore:mutex", "Failed to create/open named mutex");

    result.hEvent = CreateEventA(NULL, TRUE, FALSE, ename);
    if (!result.hEvent)
        mexErrMsgIdAndTxt("semaphore:event", "Failed to create/open named event");

    WaitForSingleObject(result.hMutex, INFINITE);
    return result;
}

shared_semaphore_t* create_semaphore(int key, int initval, int* out_count) {
    semaphore_map_result_t mapped = map_shared_semaphore(key, TRUE);

    if (!mapped.sem->initialized || mapped.is_new) {
        ZeroMemory(mapped.sem, sizeof(shared_semaphore_t));
        mapped.sem->count = initval;
        mapped.sem->max = initval;
        mapped.sem->initialized = 1;
    } else {
        mapped.sem->count = initval;
        mapped.sem->terminate = 0;
        SetEvent(mapped.hEvent);
    }

    if (out_count) *out_count = mapped.sem->count;
    ReleaseMutex(mapped.hMutex);
    return mapped.sem;
}

semaphore_map_result_t get_semaphore(int key) {
    semaphore_map_result_t mapped = map_shared_semaphore(key, FALSE);

    if (!mapped.sem->initialized) {
        ReleaseMutex(mapped.hMutex);
        close_handles(&mapped);
        mexErrMsgIdAndTxt("semaphore:uninitialized", "Semaphore %d exists but is not initialized.", key);
    }

    return mapped;
}

static int wait_semaphore(int key) {
    semaphore_map_result_t mapped = get_semaphore(key);

    if (mapped.sem->terminate) {
        ReleaseMutex(mapped.hMutex);
        close_handles(&mapped);
        mexErrMsgIdAndTxt("semaphore:terminated", "Semaphore %d has been destroyed.", key);
    }

    int current_count = -1;

    while (1) {
        if (mapped.sem->terminate) {
            ReleaseMutex(mapped.hMutex);
            close_handles(&mapped);
            mexErrMsgIdAndTxt("semaphore:terminated", "Semaphore %d has been destroyed.", key);
        }

        if (mapped.sem->count > 0) {
            mapped.sem->count--;
            current_count = mapped.sem->count;
            if (mapped.sem->count == 0)
                ResetEvent(mapped.hEvent);
            ReleaseMutex(mapped.hMutex);
            close_handles(&mapped);
            return current_count;
        }

        ReleaseMutex(mapped.hMutex);
        WaitForSingleObject(mapped.hEvent, INFINITE);
        WaitForSingleObject(mapped.hMutex, INFINITE);
    }
}

static int post_semaphore(int key) {
    semaphore_map_result_t mapped = get_semaphore(key);

    if (mapped.sem->terminate) {
        ReleaseMutex(mapped.hMutex);
        close_handles(&mapped);
        mexErrMsgIdAndTxt("semaphore:terminated", "Semaphore %d has been destroyed.", key);
    }

    if (mapped.sem->count >= mapped.sem->max) {
        mexWarnMsgIdAndTxt("semaphore:post",
            "Cannot post: semaphore %d already at maximum count (%d).", key, mapped.sem->max);
    } else {
        mapped.sem->count++;
    }

    int current_count = mapped.sem->count;
    SetEvent(mapped.hEvent);
    ReleaseMutex(mapped.hMutex);
    close_handles(&mapped);
    return current_count;
}


static void destroy_semaphore(int key) {
    semaphore_map_result_t mapped = get_semaphore(key);
    mapped.sem->terminate = 1;
    SetEvent(mapped.hEvent);
    ReleaseMutex(mapped.hMutex);
    close_handles(&mapped);
}

#else  // POSIX

#define SHM_NAME_PREFIX_POSIX "/matlab_mex_sem_"

static shared_semaphore_t* map_posix_semaphore(int key, int create, int* is_new, int* fd_out) {
    char name[64];
    snprintf(name, sizeof(name), SHM_NAME_PREFIX_POSIX SEMAPHORE_KEY_NAME_FMT, key);

    int fd = shm_open(name, create ? (O_CREAT | O_RDWR) : O_RDWR, 0666);
    if (fd == -1)
        mexErrMsgIdAndTxt("semaphore:shm_open", "Failed to open shared memory for key %d", key);

    if (create) ftruncate(fd, sizeof(shared_semaphore_t));
    void* addr = mmap(NULL, sizeof(shared_semaphore_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED)
        mexErrMsgIdAndTxt("semaphore:mmap", "Failed to map shared memory for key %d", key);

    *is_new = create;
    *fd_out = fd;
    return (shared_semaphore_t*)addr;
}

static void ensure_mutex_locked(shared_semaphore_t* sem, int fd) {
    int err = pthread_mutex_lock(&sem->mutex);

    if (err == EOWNERDEAD) {
        pthread_mutex_consistent(&sem->mutex);
    } else if (err == EINVAL) {
        // Auto-reset corrupted semaphore
        memset(sem, 0, sizeof(shared_semaphore_t));
        pthread_mutexattr_t mattr;
        pthread_condattr_t cattr;

        pthread_mutexattr_init(&mattr);
        pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
        int robust_result = pthread_mutexattr_setrobust(&mattr, PTHREAD_MUTEX_ROBUST);
        if (robust_result != 0) {
            mexPrintf("pthread_mutexattr_setrobust failed with code %d\n", robust_result);
            mexWarnMsgIdAndTxt("semaphore:robust", "PTHREAD_MUTEX_ROBUST not supported. Crash recovery disabled.");
        }

        pthread_condattr_init(&cattr);
        pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED);

        pthread_mutex_init(&sem->mutex, &mattr);
        pthread_cond_init(&sem->cond, &cattr);

        sem->count = 0;
        sem->max = 1;
        sem->initialized = 1;
        sem->terminate = 0;

        pthread_mutex_lock(&sem->mutex); // Lock again after reinit
    } else if (err != 0) {
        close(fd);
        mexErrMsgIdAndTxt("semaphore:mutex_lock", "Failed to lock mutex: %d", err);
    }
}

shared_semaphore_t* create_semaphore(int key, int initval, int* out_count) {
    int is_new = 0, fd = -1;
    shared_semaphore_t* sem = map_posix_semaphore(key, 1, &is_new, &fd);

    flock(fd, LOCK_EX);

    // Defensive reinit if semaphore was forcibly reset (e.g., in destroy)
    if (!sem->initialized) {
        memset(sem, 0, sizeof(shared_semaphore_t));
        sem->count = initval;
        sem->max = initval;

        pthread_mutexattr_t mattr;
        pthread_condattr_t cattr;

        pthread_mutexattr_init(&mattr);
        pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
        int robust_result = pthread_mutexattr_setrobust(&mattr, PTHREAD_MUTEX_ROBUST);
        if (robust_result != 0) {
            mexPrintf("pthread_mutexattr_setrobust failed with code %d\n", robust_result);
            mexWarnMsgIdAndTxt("semaphore:robust", "PTHREAD_MUTEX_ROBUST not supported. Crash recovery disabled.");
        }

        pthread_condattr_init(&cattr);
        pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED);

        pthread_mutex_init(&sem->mutex, &mattr);
        pthread_cond_init(&sem->cond, &cattr);

        sem->initialized = 1;
    } else {
        sem->count = initval;
        sem->terminate = 0;
        pthread_cond_broadcast(&sem->cond);
    }

    flock(fd, LOCK_UN);
    if (out_count) *out_count = sem->count;
    close(fd);
    return sem;
}


shared_semaphore_t* get_semaphore(int key) {
    int is_new = 0, fd = -1;
    shared_semaphore_t* sem = map_posix_semaphore(key, 0, &is_new, &fd);

    if (!sem->initialized) {
        close(fd);
        mexErrMsgIdAndTxt("semaphore:uninitialized", "Semaphore %d exists but is not initialized.", key);
    }

    close(fd);
    return sem;
}

static int wait_semaphore(int key) {
    int fd = -1;
    shared_semaphore_t* sem = map_posix_semaphore(key, 0, &(int){0}, &fd);
    ensure_mutex_locked(sem, fd);

    if (sem->terminate) {
        pthread_mutex_unlock(&sem->mutex);
        close(fd);
        mexErrMsgIdAndTxt("semaphore:terminated", "Semaphore %d has been destroyed.", key);
    }

    while (sem->count <= 0 && !sem->terminate)
        pthread_cond_wait(&sem->cond, &sem->mutex);

    if (sem->terminate) {
        pthread_mutex_unlock(&sem->mutex);
        close(fd);
        mexErrMsgIdAndTxt("semaphore:terminated", "Semaphore %d has been destroyed.", key);
    }

    sem->count--;
    int count = sem->count;
    pthread_mutex_unlock(&sem->mutex);
    close(fd);
    return count;
}

static int post_semaphore(int key) {
    int fd = -1;
    shared_semaphore_t* sem = map_posix_semaphore(key, 0, &(int){0}, &fd);
    ensure_mutex_locked(sem, fd);

    if (sem->terminate) {
        pthread_mutex_unlock(&sem->mutex);
        close(fd);
        mexErrMsgIdAndTxt("semaphore:terminated", "Semaphore %d has been destroyed.", key);
    }

    if (sem->count >= sem->max) {
        mexWarnMsgIdAndTxt("semaphore:post",
            "Cannot post: semaphore %d already at maximum count (%d).", key, sem->max);
    } else {
        sem->count++;
    }

    int count = sem->count;
    pthread_cond_signal(&sem->cond);
    pthread_mutex_unlock(&sem->mutex);
    close(fd);
    return count;
}

static void destroy_semaphore(int key) {
    int fd = -1;
    shared_semaphore_t* sem = map_posix_semaphore(key, 0, &(int){0}, &fd);

    // Try to lock, but don't block forever.
    int err = pthread_mutex_trylock(&sem->mutex);

    if (err == 0 || err == EOWNERDEAD) {
        if (err == EOWNERDEAD) {
            pthread_mutex_consistent(&sem->mutex);
        }
        sem->terminate = 1;
        pthread_cond_broadcast(&sem->cond);
        pthread_mutex_unlock(&sem->mutex);
    } else {
        // Assume the mutex is corrupted or stuck
        mexWarnMsgIdAndTxt("semaphore:destroy", "Could not lock semaphore %d (err %d). Forcing full reset.", key, err);

        // Attempt to destroy what we can
        pthread_mutex_destroy(&sem->mutex);
        pthread_cond_destroy(&sem->cond);

        memset(sem, 0, sizeof(shared_semaphore_t));

        pthread_mutexattr_t mattr;
        pthread_condattr_t cattr;

        pthread_mutexattr_init(&mattr);
        pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
        pthread_mutexattr_setrobust(&mattr, PTHREAD_MUTEX_ROBUST);

        pthread_condattr_init(&cattr);
        pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED);

        pthread_mutex_init(&sem->mutex, &mattr);
        pthread_cond_init(&sem->cond, &cattr);

        sem->terminate = 1;
        sem->initialized = 1;
        pthread_cond_broadcast(&sem->cond);
    }

    close(fd);

    char name[64];
    snprintf(name, sizeof(name), SHM_NAME_PREFIX_POSIX SEMAPHORE_KEY_NAME_FMT, key);

    if (shm_unlink(name) != 0) {
        mexWarnMsgIdAndTxt("semaphore:shm_unlink", "shm_unlink failed for %s: %s", name, strerror(errno));
    }
}

#endif  // End of POSIX section

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs < 2)
        mexErrMsgTxt("Requires at least 2 inputs: directive and key.");

    char directive[16];
    if (mxGetString(prhs[0], directive, sizeof(directive)) != 0)
        mexErrMsgTxt("Failed to parse directive string.");

    directive[0] = (char)tolower(directive[0]);
    int key = (int)(mxGetScalar(prhs[1]) + 0.5);

    int count = -1;

    switch (directive[0]) {
        case 'c': {
            int initval = 1;
            if (nrhs > 2)
                initval = (int)(mxGetScalar(prhs[2]) + 0.5);
            create_semaphore(key, initval, &count);
            break;
        }
        case 'w':
            count = wait_semaphore(key);
            break;
        case 'p':
            count = post_semaphore(key);
            break;
        case 'd':
            destroy_semaphore(key);
            break;
        default:
            mexErrMsgTxt("Unknown directive. Use 'create', 'wait', 'post', or 'destroy'.");
    }

    if (nlhs > 0) {
        if (count >= 0)
            plhs[0] = mxCreateDoubleScalar((double)count);
        else
            plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
    }
}
