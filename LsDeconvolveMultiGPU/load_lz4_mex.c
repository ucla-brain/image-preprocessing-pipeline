/*==============================================================================
  load_lz4_mex.c
  ------------------------------------------------------------------------------
  High-performance LZ4 Chunked Loader for Large MATLAB Arrays (>2GB)

  Author:       Keivan Moradi
  Review:       ChatGPT-4.1 (2025-06)
  License:      GPL v3 (https://www.gnu.org/licenses/gpl-3.0.html)
  ------------------------------------------------------------------------------
  OVERVIEW
  --------
  This MEX function loads compressed data files written by `save_lz4_mex.c`.
  It supports arbitrarily large N-dimensional arrays (e.g., 3D, 4D, etc.)
  compressed using LZ4 and chunked to bypass MATLAB's 2GB memory limits.

  Key Features:
    • Chunked format with header and metadata
    • Supports `single`, `double`, and `uint16` MATLAB types
    • Large-array safe: handles >2GB decompressed size
    • Ensures memory and file handle cleanup on all error paths
    • Validates header, dimensions, chunk sizes, and decompression success

  FILE FORMAT
  -----------
  The on-disk binary format consists of:
    • Magic number (uint32) for file identity
    • dtype + ndims + dims[] describing the output array
    • Chunk size and number of chunks
    • Per-chunk compressed and uncompressed byte sizes
    • Appended compressed data for each chunk

  COMPILATION
  -----------
  Requires:
    - lz4.c / lz4.h (from official LZ4 library)
    - MATLAB MEX compiler

  Example (MATLAB):
    >> out = load_lz4_mex('volume.lz4c');

  NOTES
  -----
  • All memory is allocated via MATLAB APIs (`mxMalloc`, `mxCreateNumericArray`)
  • Cleaned up using macros to ensure safe exit on error
  • Total header size is fixed at 33,280 bytes and must match `save_lz4_mex.c`

==============================================================================*/


#include "mex.h"
#include "lz4.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// --- CONSTANTS ---
#define MAX_DIMS 16
#define MAX_CHUNKS 2048
#define HEADER_SIZE 33280  // Must match save_lz4_mex.c

// --- FILE FORMAT ---
#define MAGIC_NUMBER 0x4C5A4331  // 'LZ4C' (chunked)

enum dtype_enum { DT_DOUBLE = 1, DT_SINGLE = 2, DT_UINT16 = 3 };

typedef struct {
    uint32_t magic;
    uint8_t dtype;
    uint8_t ndims;
    uint64_t dims[MAX_DIMS];
    uint64_t total_uncompressed;
    uint64_t chunk_size;
    uint32_t num_chunks;
    uint64_t chunk_uncomp[MAX_CHUNKS];
    uint64_t chunk_comp[MAX_CHUNKS];
    uint8_t padding[HEADER_SIZE - (4 + 1 + 1 + 8 * MAX_DIMS + 8 + 8 + 4 + 8 * MAX_CHUNKS * 2)];
} file_header_t;

int read_header(FILE* f, file_header_t* hdr) {
    size_t got = fread(hdr, 1, HEADER_SIZE, f);
    if (got != HEADER_SIZE)
        return 0;
    if (hdr->magic != MAGIC_NUMBER)
        return 0;
    return 1;
}

// Cleanup macro to ensure file handle is closed on error
#define SAFE_FCLOSE(fp) if ((fp) != NULL) { fclose(fp); (fp) = NULL; }
#define MEX_ERROR_CLEANUP(msg) do { SAFE_FCLOSE(f); mexErrMsgTxt(msg); } while(0)

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 1)
        mexErrMsgTxt("Usage: arr = load_lz4_mex(filename)");

    // --- Open file ---
    char fname[1024];
    mxGetString(prhs[0], fname, sizeof(fname));
    FILE* f = fopen(fname, "rb");
    if (!f) mexErrMsgTxt("Failed to open file for reading.");

    // --- Read and validate header ---
    file_header_t hdr = {0};
    if (!read_header(f, &hdr))
        MEX_ERROR_CLEANUP("Failed to read or validate LZ4 header.");

    if (hdr.ndims > MAX_DIMS || hdr.num_chunks > MAX_CHUNKS)
        MEX_ERROR_CLEANUP("File metadata exceeds supported limits.");

    // --- Set MATLAB dimensions ---
    mwSize ndims = hdr.ndims;
    mwSize dims[MAX_DIMS];
    for (mwSize i = 0; i < ndims; ++i)
        dims[i] = (mwSize)hdr.dims[i];

    // --- Determine output type ---
    mxClassID classid;
    if (hdr.dtype == DT_DOUBLE) classid = mxDOUBLE_CLASS;
    else if (hdr.dtype == DT_SINGLE) classid = mxSINGLE_CLASS;
    else if (hdr.dtype == DT_UINT16) classid = mxUINT16_CLASS;
    else
        MEX_ERROR_CLEANUP("Unsupported dtype in file.");

    // --- Allocate output array ---
    mxArray* arr = mxCreateNumericArray(ndims, dims, classid, mxREAL);
    if (!arr)
        MEX_ERROR_CLEANUP("Out of memory creating output array.");

    char* dest = (char*)mxGetData(arr);
    uint64_t total_uncompressed = hdr.total_uncompressed;
    uint64_t copied = 0;

    // --- Decompress all chunks ---
    for (uint32_t i = 0; i < hdr.num_chunks; ++i)
    {
        uint64_t this_uncomp = hdr.chunk_uncomp[i];
        uint64_t this_comp = hdr.chunk_comp[i];

        if (this_uncomp > 0x7FFFFFFF || this_comp > 0x7FFFFFFF)
            MEX_ERROR_CLEANUP("LZ4: Chunk size too large. File may be corrupt.");

        int uncomp_size = (int)this_uncomp;
        int comp_size = (int)this_comp;

        char* cbuf = (char*)mxMalloc(comp_size);
        if (!cbuf)
            MEX_ERROR_CLEANUP("Out of memory during chunk buffer allocation.");

        size_t nread = fread(cbuf, 1, comp_size, f);
        if (nread != (size_t)comp_size) {
            mxFree(cbuf);
            MEX_ERROR_CLEANUP("Failed to read compressed chunk.");
        }

        int dsize = LZ4_decompress_safe(cbuf, dest + copied, comp_size, uncomp_size);
        mxFree(cbuf);

        if (dsize < 0 || dsize != uncomp_size)
            MEX_ERROR_CLEANUP("LZ4 decompression failed or file corrupt.");

        copied += this_uncomp;
        if (copied > total_uncompressed)
            MEX_ERROR_CLEANUP("Decompressed size exceeds expected total.");
    }

    SAFE_FCLOSE(f);
    plhs[0] = arr;
}
