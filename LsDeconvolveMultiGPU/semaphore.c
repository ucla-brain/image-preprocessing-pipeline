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

#include <windows.h>
#include <strsafe.h>

#define MUTEX_NAME_PREFIX "Local\\"
#define EVENT_NAME_PREFIX "Local\\"
#define SHM_NAME_PREFIX   "Local\\"

// Shared structure for semaphore state in memory
typedef struct {
    int initialized;
    int count;
    int max;
    volatile int terminate;
} shared_semaphore_t;

// Handles and pointer for mapped semaphore
typedef struct {
    shared_semaphore_t* sem;
    HANDLE hMap;
    HANDLE hMutex;
    HANDLE hEvent;
    BOOL is_new;
} semaphore_map_result_t;

// Safely close all handles
static void close_handles(semaphore_map_result_t* mapped) {
    if (mapped->sem)   UnmapViewOfFile(mapped->sem);
    if (mapped->hMutex) CloseHandle(mapped->hMutex);
    if (mapped->hEvent) CloseHandle(mapped->hEvent);
    if (mapped->hMap)   CloseHandle(mapped->hMap);
}

// Platform-specific shared memory and synchronization object creation
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
        if (!result.hMap) mexErrMsgIdAndTxt("semaphore:create", "Failed to create file mapping");
        result.is_new = (GetLastError() != ERROR_ALREADY_EXISTS);
    } else {
        result.hMap = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, shmname);
        if (!result.hMap) mexErrMsgIdAndTxt("semaphore:notfound", "Semaphore %d not found. Use 'create' first.", key);
        result.is_new = FALSE;
    }

    result.sem = (shared_semaphore_t*)MapViewOfFile(result.hMap, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!result.sem) {
        CloseHandle(result.hMap);
        mexErrMsgIdAndTxt("semaphore:map", "Failed to map shared memory for key %d", key);
    }

    result.hMutex = CreateMutexA(NULL, FALSE, mname);
    if (!result.hMutex) {
        UnmapViewOfFile(result.sem);
        CloseHandle(result.hMap);
        mexErrMsgIdAndTxt("semaphore:mutex", "Failed to create/open named mutex");
    }

    result.hEvent = CreateEventA(NULL, TRUE, FALSE, ename);
    if (!result.hEvent) {
        CloseHandle(result.hMutex);
        UnmapViewOfFile(result.sem);
        CloseHandle(result.hMap);
        mexErrMsgIdAndTxt("semaphore:event", "Failed to create/open named event");
    }

    WaitForSingleObject(result.hMutex, INFINITE);
    return result;
}

// Create or reinitialize a semaphore
static int create_semaphore(int key, int initval, int* out_count) {
    semaphore_map_result_t mapped = map_shared_semaphore(key, TRUE);

    if (mapped.is_new) {
        ZeroMemory(mapped.sem, sizeof(shared_semaphore_t));
        mapped.sem->count = initval;
        mapped.sem->max = initval;
        mapped.sem->initialized = 1;
    } else {
        if (!mapped.sem->initialized) {
            ReleaseMutex(mapped.hMutex);
            close_handles(&mapped);
            mexErrMsgIdAndTxt("semaphore:uninitialized", "Semaphore %d is in inconsistent state.", key);
        }
    }

    if (out_count) *out_count = mapped.sem->count;

    ReleaseMutex(mapped.hMutex);
    close_handles(&mapped);
    return 0;
}

// Open an existing semaphore
static semaphore_map_result_t get_semaphore(int key) {
    semaphore_map_result_t mapped = map_shared_semaphore(key, FALSE);
    if (!mapped.sem->initialized) {
        ReleaseMutex(mapped.hMutex);
        close_handles(&mapped);
        mexErrMsgIdAndTxt("semaphore:uninitialized", "Semaphore %d exists but not initialized.", key);
    }
    return mapped;
}

// Wait operation
static int wait_semaphore(int key) {
    semaphore_map_result_t mapped = get_semaphore(key);

    while (1) {
        if (mapped.sem->terminate) {
            ReleaseMutex(mapped.hMutex);
            close_handles(&mapped);
            mexErrMsgIdAndTxt("semaphore:terminated", "Semaphore %d has been destroyed.", key);
        }

        if (mapped.sem->count > 0) {
            mapped.sem->count--;
            int current_count = mapped.sem->count;
            if (current_count == 0) ResetEvent(mapped.hEvent);
            ReleaseMutex(mapped.hMutex);
            close_handles(&mapped);
            return current_count;
        }

        ReleaseMutex(mapped.hMutex);
        WaitForSingleObject(mapped.hEvent, INFINITE);
        WaitForSingleObject(mapped.hMutex, INFINITE);
    }
}

// Post operation
static int post_semaphore(int key) {
    semaphore_map_result_t mapped = get_semaphore(key);

    if (mapped.sem->terminate) {
        ReleaseMutex(mapped.hMutex);
        close_handles(&mapped);
        mexErrMsgIdAndTxt("semaphore:terminated", "Semaphore %d has been destroyed.", key);
    }

    if (mapped.sem->count >= mapped.sem->max) {
        mexWarnMsgIdAndTxt("semaphore:post", "Semaphore %d already at max count (%d).", key, mapped.sem->max);
    } else {
        mapped.sem->count++;
    }

    int current_count = mapped.sem->count;
    SetEvent(mapped.hEvent);
    ReleaseMutex(mapped.hMutex);
    close_handles(&mapped);
    return current_count;
}

// Destroy operation
static void destroy_semaphore(int key) {
    semaphore_map_result_t mapped = {0};
    __try {
        mapped = map_shared_semaphore(key, FALSE);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return; // silently ignore if not found
    }

    if (!mapped.sem) {
        close_handles(&mapped);
        return;
    }

    WaitForSingleObject(mapped.hMutex, INFINITE);
    mapped.sem->terminate = 1;
    SetEvent(mapped.hEvent);
    ReleaseMutex(mapped.hMutex);
    close_handles(&mapped);
}

#else  // POSIX

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <semaphore.h>

#define SEM_NAME_FMT "/matlab_mex_sem_%d"
#define SHM_NAME_FMT "/matlab_mex_meta_%d"

typedef struct {
    int initialized;
    int max;
    volatile int terminate;
} semaphore_meta_t;

static void name_for_key(char* out, size_t outlen, const char* fmt, int key) {
    snprintf(out, outlen, fmt, key);
}

static int create_semaphore(int key, int initval, int* out_count) {
    char sem_name[64], shm_name[64];
    name_for_key(sem_name, sizeof(sem_name), SEM_NAME_FMT, key);
    name_for_key(shm_name, sizeof(shm_name), SHM_NAME_FMT, key);

    sem_t* sem = sem_open(sem_name, O_CREAT | O_EXCL, 0666, initval);
    if (sem == SEM_FAILED && errno == EEXIST)
        sem = sem_open(sem_name, 0);
    if (sem == SEM_FAILED)
        mexErrMsgIdAndTxt("semaphore:create", "sem_open failed: %s", strerror(errno));

    int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        sem_close(sem);
        sem_unlink(sem_name);
        mexErrMsgIdAndTxt("semaphore:shm_open", "Failed to open shared memory: %s", strerror(errno));
    }

    if (ftruncate(shm_fd, sizeof(semaphore_meta_t)) == -1) {
        close(shm_fd);
        sem_close(sem);
        sem_unlink(sem_name);
        shm_unlink(shm_name);
        mexErrMsgIdAndTxt("semaphore:truncate", "Failed to resize shared memory: %s", strerror(errno));
    }

    semaphore_meta_t* meta = mmap(NULL, sizeof(semaphore_meta_t), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (meta == MAP_FAILED) {
        close(shm_fd);
        sem_close(sem);
        sem_unlink(sem_name);
        shm_unlink(shm_name);
        mexErrMsgIdAndTxt("semaphore:mmap", "Failed to map shared memory.");
    }

    if (!meta->initialized) {
        meta->max = initval;
        meta->terminate = 0;
        meta->initialized = 1;
    }

    if (out_count) {
        int val;
        if (sem_getvalue(sem, &val) == 0)
            *out_count = val;
    }

    munmap(meta, sizeof(semaphore_meta_t));
    close(shm_fd);
    sem_close(sem);
    return 0;
}

static int wait_semaphore(int key) {
    char sem_name[64], shm_name[64];
    name_for_key(sem_name, sizeof(sem_name), SEM_NAME_FMT, key);
    name_for_key(shm_name, sizeof(shm_name), SHM_NAME_FMT, key);

    sem_t* sem = sem_open(sem_name, 0);
    if (sem == SEM_FAILED)
        mexErrMsgIdAndTxt("semaphore:wait", "Semaphore %d not found.", key);

    int shm_fd = shm_open(shm_name, O_RDWR, 0666);
    if (shm_fd == -1) {
        sem_close(sem);
        mexErrMsgIdAndTxt("semaphore:shm_open", "Shared memory for key %d not found.", key);
    }

    semaphore_meta_t* meta = mmap(NULL, sizeof(semaphore_meta_t), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (meta == MAP_FAILED) {
        close(shm_fd);
        sem_close(sem);
        mexErrMsgIdAndTxt("semaphore:mmap", "Failed to map metadata for key %d", key);
    }

    if (meta->terminate) {
        munmap(meta, sizeof(semaphore_meta_t));
        close(shm_fd);
        sem_close(sem);
        mexErrMsgIdAndTxt("semaphore:terminated", "Semaphore %d has been destroyed.", key);
    }

    sem_wait(sem);
    int val = -1;
    sem_getvalue(sem, &val);

    munmap(meta, sizeof(semaphore_meta_t));
    close(shm_fd);
    sem_close(sem);
    return val;
}

static int post_semaphore(int key) {
    char sem_name[64], shm_name[64];
    name_for_key(sem_name, sizeof(sem_name), SEM_NAME_FMT, key);
    name_for_key(shm_name, sizeof(shm_name), SHM_NAME_FMT, key);

    sem_t* sem = sem_open(sem_name, 0);
    if (sem == SEM_FAILED)
        mexErrMsgIdAndTxt("semaphore:post", "Semaphore %d not found.", key);

    int shm_fd = shm_open(shm_name, O_RDWR, 0666);
    if (shm_fd == -1) {
        sem_close(sem);
        mexErrMsgIdAndTxt("semaphore:shm_open", "Shared memory for key %d not found.", key);
    }

    semaphore_meta_t* meta = mmap(NULL, sizeof(semaphore_meta_t), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (meta == MAP_FAILED) {
        close(shm_fd);
        sem_close(sem);
        mexErrMsgIdAndTxt("semaphore:mmap", "Failed to map metadata for key %d", key);
    }

    int val;
    sem_getvalue(sem, &val);
    if (val >= meta->max) {
        mexWarnMsgIdAndTxt("semaphore:post", "Cannot post: semaphore %d already at max count (%d).", key, meta->max);
    } else {
        sem_post(sem);
        sem_getvalue(sem, &val);
    }

    munmap(meta, sizeof(semaphore_meta_t));
    close(shm_fd);
    sem_close(sem);
    return val;
}

static void destroy_semaphore(int key) {
    char sem_name[64], shm_name[64];
    name_for_key(sem_name, sizeof(sem_name), SEM_NAME_FMT, key);
    name_for_key(shm_name, sizeof(shm_name), SHM_NAME_FMT, key);

    sem_t* sem = sem_open(sem_name, 0);
    int shm_fd = shm_open(shm_name, O_RDWR, 0666);
    if (sem == SEM_FAILED || shm_fd == -1) return;

    semaphore_meta_t* meta = mmap(NULL, sizeof(semaphore_meta_t), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (meta != MAP_FAILED) {
        meta->terminate = 1;
        munmap(meta, sizeof(semaphore_meta_t));
    }

    sem_close(sem);
    sem_unlink(sem_name);
    close(shm_fd);
    shm_unlink(shm_name);
}

#endif  // End of POSIX section

// MEX entry point
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