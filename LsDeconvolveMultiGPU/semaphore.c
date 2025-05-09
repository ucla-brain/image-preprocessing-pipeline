/*
 * Copyright (c) 2011 Joshua V Dillon
 * Copyright (c) 2014 Andrew Smart (besed on "semaphore.c" by Joshua V. Dillon)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the
 * following conditions are met:
 *  * Redistributions of source code must retain the above
 *    copyright notice, this list of conditions and the
 *    following disclaimer.
 *  * Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the
 *    following disclaimer in the documentation and/or other
 *    materials provided with the distribution.
 *  * Neither the name of the author nor the names of its
 *    contributors may be used to endorse or promote products
 *    derived from this software without specific prior written
 *    permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL JOSHUA
 * V DILLON BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */


/*
 * This code can be compiled from within Matlab or command-line, assuming the
 * system is appropriately setup.  To compile, invoke:
 *
 * For 32-bit machines:
 *     mex -O -v semaphore.c
 * For 64-bit machines:
 *     mex -O -v semaphore.c
 *
 */


/* 
 * Programmer's Notes:
 *
 * MEX C API:
 * http://www.mathworks.com/access/helpdesk/help/techdoc/apiref/bqoqnz0.html
 *
 * Testing:
 *
 */

/*
 * Updated cross-platform MATLAB semaphore implementation (2025)
 * Incorporates fixes for validation, error handling, Windows handle leaks, and safer shared memory usage.
 * Author: Based on code by Joshua V Dillon and Andrew Smart
 * Modifications: OpenAI GPT-4, 2025
 */

#ifdef _CRT_SECURE_NO_DEPRECATE
  #define WIN32
#endif

#ifndef WIN32
  #include <sys/shm.h>
  #include <semaphore.h>
  #include <unistd.h>
#else
  #include <windows.h>
  #include <stdio.h>
#endif

#include <errno.h>
#include <string.h>
#include "mex.h"

#define MAXDIRECTIVELEN 256
#define SEM_MAGIC 0xA5A5BEEF  // Integrity check for shared memory

#ifndef WIN32
typedef struct {
  int magic;
  sem_t sem;
} shared_semaphore_t;
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  char directive[MAXDIRECTIVELEN + 1];
  int semval = 1;

#ifndef WIN32
  key_t semkey = 0;
  int semid;
  int semflg = 0644;
  shared_semaphore_t *shm = NULL;
#else
  char semkeyStr[MAXDIRECTIVELEN];
  HANDLE hSemaphore = NULL;
  DWORD lastError;
  LPVOID lpMsgBuf;
#endif

  if (nrhs < 2) mexErrMsgIdAndTxt("MATLAB:semaphore", "Minimum input arguments missing; must supply directive and key.");
  if (mxGetString(prhs[0], directive, MAXDIRECTIVELEN) != 0)
    mexErrMsgIdAndTxt("MATLAB:semaphore", "First input argument must be one of {'create','wait','post','destroy'}.");
  if (mxGetNumberOfElements(prhs[1]) != 1 || !mxIsNumeric(prhs[1]))
    mexErrMsgIdAndTxt("MATLAB:semaphore", "Second input argument must be a valid integral key.");

#ifndef WIN32
  semkey = (key_t)(mxGetScalar(prhs[1]) + 0.5);
#else
  if (sprintf(semkeyStr, "%d", (int)(mxGetScalar(prhs[1]) + 0.5)) < 0)
    mexErrMsgIdAndTxt("MATLAB:semaphore", "Second input argument must be a valid integral key.");
#endif

  if (nlhs > 1) mexErrMsgIdAndTxt("MATLAB:semaphore", "Function returns only one value.");

  switch (tolower(directive[0])) {
    case 'c':  // Create
#ifndef WIN32
      if (nrhs > 2 && mxIsNumeric(prhs[2]) && mxGetNumberOfElements(prhs[2]) == 1)
        semval = (int)(mxGetScalar(prhs[2]) + 0.5);
      else
        mexErrMsgIdAndTxt("MATLAB:semaphore:create", "Third input argument must be initial semaphore value (numeric scalar).");

      semflg |= IPC_CREAT | IPC_EXCL;
      semid = shmget(semkey, sizeof(shared_semaphore_t), semflg);
      if (semid < 0) mexErrMsgIdAndTxt("MATLAB:semaphore:create", "Unable to create shared memory segment.");

      shm = (shared_semaphore_t *)shmat(semid, NULL, 0);
      if (shm == (void *)-1) mexErrMsgIdAndTxt("MATLAB:semaphore:create", "Unable to attach shared memory to data space.");

      shm->magic = SEM_MAGIC;
      if (sem_init(&(shm->sem), 1, semval) < 0) mexErrMsgIdAndTxt("MATLAB:semaphore:create", "Unable to create semaphore.");

      plhs[0] = mxCreateDoubleScalar(1.0);
#else
      hSemaphore = CreateSemaphore(NULL, semval, semval, semkeyStr);
      if (hSemaphore == NULL) {
        lastError = GetLastError();
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                      NULL, lastError, 0, (LPTSTR)&lpMsgBuf, 0, NULL);
        mexErrMsgIdAndTxt("MATLAB:semaphore:create", "CreateSemaphore failed: %s", (LPCTSTR)lpMsgBuf);
        LocalFree(lpMsgBuf);
      }
      CloseHandle(hSemaphore);
#endif
      break;

    case 'w':  // Wait
#ifndef WIN32
      semid = shmget(semkey, sizeof(shared_semaphore_t), semflg);
      if (semid < 0) mexErrMsgIdAndTxt("MATLAB:semaphore:wait", "Unable to locate shared memory segment.");

      shm = (shared_semaphore_t *)shmat(semid, NULL, 0);
      if (shm == (void *)-1) mexErrMsgIdAndTxt("MATLAB:semaphore:wait", "Unable to attach shared memory.");
      if (shm->magic != SEM_MAGIC) mexErrMsgIdAndTxt("MATLAB:semaphore:wait", "Invalid semaphore memory detected.");

      while (sem_wait(&(shm->sem)) != 0) {
        if (errno != EINTR)
          mexErrMsgIdAndTxt("MATLAB:semaphore:wait", "Error waiting on semaphore: %s", strerror(errno));
      }
      shmdt(shm);
      plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
#else
      hSemaphore = OpenSemaphore(SYNCHRONIZE, FALSE, semkeyStr);
      if (hSemaphore == NULL)
        mexErrMsgIdAndTxt("MATLAB:semaphore:wait", "Unable to open semaphore.");
      if (WaitForSingleObject(hSemaphore, INFINITE) == WAIT_FAILED)
        mexErrMsgIdAndTxt("MATLAB:semaphore:wait", "Wait failed.");
      CloseHandle(hSemaphore);
#endif
      break;

    case 'p':  // Post
#ifndef WIN32
      semid = shmget(semkey, sizeof(shared_semaphore_t), semflg);
      if (semid < 0) mexErrMsgIdAndTxt("MATLAB:semaphore:post", "Unable to locate shared memory segment.");

      shm = (shared_semaphore_t *)shmat(semid, NULL, 0);
      if (shm == (void *)-1) mexErrMsgIdAndTxt("MATLAB:semaphore:post", "Unable to attach shared memory.");
      if (shm->magic != SEM_MAGIC) mexErrMsgIdAndTxt("MATLAB:semaphore:post", "Invalid semaphore memory detected.");

      if (sem_post(&(shm->sem)) < 0)
        mexErrMsgIdAndTxt("MATLAB:semaphore:post", "Failed to post semaphore: %s", strerror(errno));

      shmdt(shm);
      plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
#else
      hSemaphore = OpenSemaphore(SEMAPHORE_MODIFY_STATE, FALSE, semkeyStr);
      if (hSemaphore == NULL)
        mexErrMsgIdAndTxt("MATLAB:semaphore:post", "Unable to open semaphore.");
      if (!ReleaseSemaphore(hSemaphore, 1, NULL)) {
        lastError = GetLastError();
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                      NULL, lastError, 0, (LPTSTR)&lpMsgBuf, 0, NULL);
        mexErrMsgIdAndTxt("MATLAB:semaphore:post", "ReleaseSemaphore failed: %s", (LPCTSTR)lpMsgBuf);
        LocalFree(lpMsgBuf);
      }
      CloseHandle(hSemaphore);
#endif
      break;

    case 'd':  // Destroy
#ifndef WIN32
      semid = shmget(semkey, sizeof(shared_semaphore_t), semflg);
      if (semid < 0) mexErrMsgIdAndTxt("MATLAB:semaphore:destroy", "Unable to locate shared memory segment.");

      shm = (shared_semaphore_t *)shmat(semid, NULL, 0);
      if (shm == (void *)-1) mexErrMsgIdAndTxt("MATLAB:semaphore:destroy", "Unable to attach shared memory.");
      if (shm->magic != SEM_MAGIC) mexErrMsgIdAndTxt("MATLAB:semaphore:destroy", "Invalid semaphore memory detected.");

      if (sem_destroy(&(shm->sem)) < 0)
        mexErrMsgIdAndTxt("MATLAB:semaphore:destroy", "Unable to destroy semaphore: %s", strerror(errno));

      shm->magic = 0;
      shmdt(shm);
      shmctl(semid, IPC_RMID, NULL);
      plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
#else
      // No-op on Windows
#endif
      break;

    default:
      mexErrMsgIdAndTxt("MATLAB:semaphore", "Unrecognized directive.");
  }
}
