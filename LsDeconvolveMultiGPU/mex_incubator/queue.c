#include "mex.h"
#include "matrix.h"

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/ipc.h>
    #include <sys/shm.h>
    #include <pthread.h>
    #include <unistd.h>
#endif

#include <string.h>
#include <stdio.h>

#define MAX_QUEUE_SIZE 1024
#define MUTEX_NAME_PREFIX "queue_mutex_"
#define MEM_NAME_PREFIX   "queue_mem_"

typedef struct {
    int head;
    int tail;
    int count;
    double data[MAX_QUEUE_SIZE];  // Use doubles for compatibility with MATLAB
} SharedQueue;

#ifdef _WIN32
typedef struct {
    HANDLE hMapFile;
    HANDLE hMutex;
    SharedQueue* queue;
} SharedResources;
#else
typedef struct {
    int shmid;
    SharedQueue* queue;
    pthread_mutex_t* mutex;
} SharedResources;
#endif

void get_key_string(double key, char* out, const char* prefix) {
    int k = (int)(key + 0.5);
    snprintf(out, 256, "%s%d", prefix, k);
}

#ifdef _WIN32
int open_or_create_shared_memory(const char* name, HANDLE* hMapFile, SharedQueue** ptr, int create) {
    if (create)
        *hMapFile = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(SharedQueue), name);
    else
        *hMapFile = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, name);

    if (*hMapFile == NULL) return 0;

    *ptr = (SharedQueue*) MapViewOfFile(*hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedQueue));
    return (*ptr != NULL);
}

int open_or_create_mutex(const char* name, HANDLE* hMutex, int create) {
    if (create)
        *hMutex = CreateMutex(NULL, FALSE, name);
    else
        *hMutex = OpenMutex(MUTEX_ALL_ACCESS, FALSE, name);

    return (*hMutex != NULL);
}

void acquire_mutex(HANDLE hMutex) {
    WaitForSingleObject(hMutex, INFINITE);
}

void release_mutex(HANDLE hMutex) {
    ReleaseMutex(hMutex);
}
#else
int open_or_create_shared_memory(key_t key, SharedResources* res, int create) {
    int flags = 0666 | (create ? IPC_CREAT : 0);
    res->shmid = shmget(key, sizeof(SharedQueue) + sizeof(pthread_mutex_t), flags);
    if (res->shmid < 0) return 0;

    void* mem = shmat(res->shmid, NULL, 0);
    if (mem == (void*)-1) return 0;

    res->queue = (SharedQueue*)mem;
    res->mutex = (pthread_mutex_t*)((char*)mem + sizeof(SharedQueue));

    if (create) {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(res->mutex, &attr);
    }

    return 1;
}

void acquire_mutex(pthread_mutex_t* m) {
    pthread_mutex_lock(m);
}

void release_mutex(pthread_mutex_t* m) {
    pthread_mutex_unlock(m);
}
#endif

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 2)
        mexErrMsgIdAndTxt("queue:minArgs", "Need at least 2 arguments: directive and key.");

    char directive[16] = {0};
    if (mxGetString(prhs[0], directive, sizeof(directive)) != 0)
        mexErrMsgIdAndTxt("queue:invalidDirective", "Failed to parse directive string.");

    double key = mxGetScalar(prhs[1]);

    SharedResources res = { 0 };

#ifdef _WIN32
    char memName[256], mutexName[256];
    get_key_string(key, memName, MEM_NAME_PREFIX);
    get_key_string(key, mutexName, MUTEX_NAME_PREFIX);
#else
    key_t ipc_key = (key_t)(key + 0.5);
#endif

    if (directive[0] == 'c') {
        if (nrhs < 3 || !mxIsDouble(prhs[2]))
            mexErrMsgIdAndTxt("queue:create", "Third argument must be a vector of doubles.");

        size_t len = mxGetNumberOfElements(prhs[2]);
        double* data = mxGetPr(prhs[2]);

#ifdef _WIN32
        if (!open_or_create_shared_memory(memName, &res.hMapFile, &res.queue, 1))
            mexErrMsgIdAndTxt("queue:create", "Failed to create shared memory.");
        if (!open_or_create_mutex(mutexName, &res.hMutex, 1))
            mexErrMsgIdAndTxt("queue:create", "Failed to create mutex.");
        acquire_mutex(res.hMutex);
#else
        if (!open_or_create_shared_memory(ipc_key, &res, 1))
            mexErrMsgIdAndTxt("queue:create", "Failed to create shared memory.");
        acquire_mutex(res.mutex);
#endif

        res.queue->head = 0;
        res.queue->tail = (int)len % MAX_QUEUE_SIZE;
        res.queue->count = (int)len;
        for (int i = 0; i < len && i < MAX_QUEUE_SIZE; ++i)
            res.queue->data[i] = data[i];

#ifdef _WIN32
        release_mutex(res.hMutex);
#else
        release_mutex(res.mutex);
#endif
        plhs[0] = mxCreateDoubleScalar(1.0);
    }

    else if (directive[0] == 'p') {
        if (nrhs < 3)
            mexErrMsgIdAndTxt("queue:post", "Need value to post.");

        double val = mxGetScalar(prhs[2]);

#ifdef _WIN32
        if (!open_or_create_shared_memory(memName, &res.hMapFile, &res.queue, 0))
            mexErrMsgIdAndTxt("queue:post", "Failed to open shared memory.");
        if (!open_or_create_mutex(mutexName, &res.hMutex, 0))
            mexErrMsgIdAndTxt("queue:post", "Failed to open mutex.");
        acquire_mutex(res.hMutex);
#else
        if (!open_or_create_shared_memory(ipc_key, &res, 0))
            mexErrMsgIdAndTxt("queue:post", "Failed to open shared memory.");
        acquire_mutex(res.mutex);
#endif

        if (res.queue->count < MAX_QUEUE_SIZE) {
            res.queue->data[res.queue->tail] = val;
            res.queue->tail = (res.queue->tail + 1) % MAX_QUEUE_SIZE;
            res.queue->count++;
        }

#ifdef _WIN32
        release_mutex(res.hMutex);
#else
        release_mutex(res.mutex);
#endif
    }

    else if (directive[0] == 'w') {
#ifdef _WIN32
        if (!open_or_create_shared_memory(memName, &res.hMapFile, &res.queue, 0))
            mexErrMsgIdAndTxt("queue:wait", "Failed to open shared memory.");
        if (!open_or_create_mutex(mutexName, &res.hMutex, 0))
            mexErrMsgIdAndTxt("queue:wait", "Failed to open mutex.");
#else
        if (!open_or_create_shared_memory(ipc_key, &res, 0))
            mexErrMsgIdAndTxt("queue:wait", "Failed to open shared memory.");
#endif

        double value = 0.0;
        while (1) {
#ifdef _WIN32
            acquire_mutex(res.hMutex);
#else
            acquire_mutex(res.mutex);
#endif
            if (res.queue->count > 0) {
                value = res.queue->data[res.queue->head];
                res.queue->head = (res.queue->head + 1) % MAX_QUEUE_SIZE;
                res.queue->count--;
#ifdef _WIN32
                release_mutex(res.hMutex);
#else
                release_mutex(res.mutex);
#endif
                break;
            }
#ifdef _WIN32
            release_mutex(res.hMutex);
            Sleep(10);
#else
            release_mutex(res.mutex);
            usleep(10000);
#endif
        }
        plhs[0] = mxCreateDoubleScalar(value);
    }

    else if (directive[0] == 'd') {
#ifdef _WIN32
        HANDLE hMap = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, memName);
        if (hMap) {
            LPVOID ptr = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedQueue));
            if (ptr) UnmapViewOfFile(ptr);
            CloseHandle(hMap);
        }

        HANDLE hMtx = OpenMutex(MUTEX_ALL_ACCESS, FALSE, mutexName);
        if (hMtx) CloseHandle(hMtx);
#else
        if (!open_or_create_shared_memory(ipc_key, &res, 0))
            mexErrMsgIdAndTxt("queue:destroy", "Failed to open shared memory.");

        shmdt(res.queue);
        shmctl(res.shmid, IPC_RMID, 0);
#endif
        plhs[0] = mxCreateDoubleScalar(1.0);
    }

    else {
        mexErrMsgIdAndTxt("queue:invalid", "Unknown directive.");
    }
}
