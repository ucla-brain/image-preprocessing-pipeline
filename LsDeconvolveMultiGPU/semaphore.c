/*
 * semaphore.c - POSIX and Windows MEX semaphore support
 *
 * Keivan Moradi (2025) @ Brain Research and Artificial Intelligence Nexus lab @ UCLA
 * - Cross-platform implementation for MATLAB MEX
 * - Uses POSIX named semaphores on Linux/macOS
 * - Uses named Win32 semaphores on Windows
 * - Clean, centralized error handling for both platforms
 *
 * Compilation:
 *   You can compile this file from the MATLAB command line using:
 *     mex -O -v semaphore.c
 *
 * References:
 *   - MATLAB MEX C API: https://www.mathworks.com/help/matlab/write-cc-mex-files.html
 *
 * License:
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include <errno.h>
#include "mex.h"

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #include <stdio.h>
    #define IS_WINDOWS 1
#else
    #include <semaphore.h>
    #include <fcntl.h>      // O_CREAT, O_EXCL
    #include <sys/stat.h>   // mode constants
    #include <unistd.h>
    #include <string.h>
    #include <ctype.h>
    #define IS_WINDOWS 0
#endif

#define MAXDIRECTIVELEN 256
#define SEM_NAME_PREFIX_WIN "Local\\LSDCONVMULTIGPU_semaphore_mem_"
#define SEM_NAME_PREFIX_POSIX "/LSDCONVMULTIGPU_semaphore_mem_"

#if IS_WINDOWS
void winLastErrorExit(const char *id, const char *msgPrefix) {
    DWORD errorCode = GetLastError();
    LPVOID lpMsgBuf = NULL;
    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_MAX_WIDTH_MASK,
        NULL, errorCode,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&lpMsgBuf, 0, NULL);

    mexErrMsgIdAndTxt(id, "%s System error #%d: \"%s\".", msgPrefix, errorCode, (LPCTSTR)lpMsgBuf);
    LocalFree(lpMsgBuf);
}
#else
void posixLastErrorExit(const char *id, const char *msgPrefix) {
    int errcode = errno;
    mexErrMsgIdAndTxt(id, "%s POSIX error #%d: \"%s\".", msgPrefix, errcode, strerror(errcode));
}
#endif

#if IS_WINDOWS
void get_key_string(double key, char* out) {
    int k = (int)(key + 0.5);
    snprintf(out, MAXDIRECTIVELEN, SEM_NAME_PREFIX_WIN "%d", k);
}
#else
void get_key_string(double key, char* out) {
    int k = (int)(key + 0.5);
    snprintf(out, MAXDIRECTIVELEN, SEM_NAME_PREFIX_POSIX "%d", k);
}
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    char directive[MAXDIRECTIVELEN + 1];
    int semval = 1;
    char semkeyStr[MAXDIRECTIVELEN];

#if IS_WINDOWS
    HANDLE hSemaphore = NULL;
#else
    sem_t *sem = NULL;
#endif

    if (nrhs < 2)
        mexErrMsgIdAndTxt("MATLAB:semaphore", "Minimum input arguments missing; must supply directive and key.");

    if (mxGetString(prhs[0], directive, MAXDIRECTIVELEN) != 0)
        mexErrMsgIdAndTxt("MATLAB:semaphore", "First input argument must be one of {'create','wait','post','destroy'}.");

    if (mxGetNumberOfElements(prhs[1]) != 1 || !mxIsNumeric(prhs[1]))
        mexErrMsgIdAndTxt("MATLAB:semaphore", "Second input argument must be a valid integral key.");

    get_key_string(mxGetScalar(prhs[1]), semkeyStr);

    if (nlhs > 1)
        mexErrMsgIdAndTxt("MATLAB:semaphore", "Function returns only one value.");

    switch (tolower(directive[0])) {
    case 'c':
        if (nrhs > 2 && mxIsNumeric(prhs[2]) && mxGetNumberOfElements(prhs[2]) == 1)
            semval = (int)(mxGetScalar(prhs[2]) + 0.5);
        else
            mexErrMsgIdAndTxt("MATLAB:semaphore:create", "Third input argument must be initial semaphore value (numeric scalar).");
#if IS_WINDOWS
        hSemaphore = CreateSemaphore(NULL, semval, semval, semkeyStr);
        if (hSemaphore == NULL)
            winLastErrorExit("MATLAB:semaphore:create", "Unable to create the semaphore.");
#else
        sem = sem_open(semkeyStr, O_CREAT | O_EXCL, 0644, semval);
        if (sem == SEM_FAILED)
            posixLastErrorExit("MATLAB:semaphore:create", "Unable to create POSIX semaphore.");
        sem_close(sem);
#endif
        plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
        break;

    case 'w':
#if IS_WINDOWS
        hSemaphore = OpenSemaphore(SYNCHRONIZE, FALSE, semkeyStr);
        if (hSemaphore == NULL)
            winLastErrorExit("MATLAB:semaphore:wait", "Unable to open the semaphore handle.");
        if (WaitForSingleObject(hSemaphore, INFINITE) == WAIT_FAILED)
            mexErrMsgIdAndTxt("MATLAB:semaphore:wait", "Semaphore wait failed.");
#else
        sem = sem_open(semkeyStr, 0);
        if (sem == SEM_FAILED)
            posixLastErrorExit("MATLAB:semaphore:wait", "Unable to open POSIX semaphore.");

        while (sem_wait(sem) != 0) {
            if (errno != EINTR)
                posixLastErrorExit("MATLAB:semaphore:wait", "Error waiting for semaphore.");
        }
        sem_close(sem);
#endif
        plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
        break;

    case 'p':
#if IS_WINDOWS
        hSemaphore = OpenSemaphore(SEMAPHORE_MODIFY_STATE, FALSE, semkeyStr);
        if (hSemaphore == NULL)
            winLastErrorExit("MATLAB:semaphore:post", "Unable to open the semaphore handle.");
        if (!ReleaseSemaphore(hSemaphore, 1, NULL))
            winLastErrorExit("MATLAB:semaphore:post", "Unable to post the semaphore.");
#else
        sem = sem_open(semkeyStr, 0);
        if (sem == SEM_FAILED)
            posixLastErrorExit("MATLAB:semaphore:post", "Unable to open POSIX semaphore.");

        if (sem_post(sem) != 0)
            posixLastErrorExit("MATLAB:semaphore:post", "Unable to post POSIX semaphore.");

        sem_close(sem);
#endif
        plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
        break;

    case 'd':
#if IS_WINDOWS
        hSemaphore = OpenSemaphore(SEMAPHORE_ALL_ACCESS, FALSE, semkeyStr);
        if (hSemaphore == NULL)
            winLastErrorExit("MATLAB:semaphore:destroy", "Unable to open semaphore handle for destruction.");
        if (!CloseHandle(hSemaphore))
            mexErrMsgIdAndTxt("MATLAB:semaphore:destroy", "Failed to close semaphore handle.");
#else
        if (sem_unlink(semkeyStr) != 0)
            posixLastErrorExit("MATLAB:semaphore:destroy", "Unable to unlink POSIX semaphore.");
#endif
        plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
        break;

    default:
        mexErrMsgIdAndTxt("MATLAB:semaphore", "Unrecognized directive.");
    }
}
