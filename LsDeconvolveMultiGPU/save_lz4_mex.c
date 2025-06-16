/*
    save_lz4_mex.c

    Chunked LZ4 compression for large (>2GB) MATLAB arrays.
    ------------------------------------------------------------------------
    Author:       Keivan Moradi (with assistance from ChatGPT v4.1, 2025)
    License:      GPL v3
    ------------------------------------------------------------------------

    Overview:
    ---------
    This MEX function saves a large MATLAB numeric array to disk using
    chunked LZ4 compression with a custom binary header. It supports
    `double`, `single`, and `uint16` arrays up to multiple terabytes in size.

    The format includes a header with metadata (datatype, dimensions, chunk
    sizes) followed by one or more compressed data chunks, each up to 1 GiB.

    This enables fast and memory-efficient I/O in multi-stage pipelines.

    Usage:
    ------
        save_lz4_mex(filename, array)

    Inputs:
        - filename : string or char array (e.g., 'data.lz4c')
        - array    : numeric MATLAB array (double, single, uint16)

    Output:
        - (none). Throws an error if saving fails.

    Notes:
    ------
        - This function is typically paired with `load_lz4_mex` for reading.
        - Chunk size is fixed at 1 GiB for compatibility and performance.
        - Header is 33 KiB and written twice (once before and once after chunking).
        - All errors trigger full cleanup to avoid leaks or corrupt files.
*/

#include "mex.h"
#include "lz4.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

// --- CONSTANTS ---
#define MAX_DIMS 16
#define MAX_CHUNKS 2048
#define HEADER_SIZE 33280  // >= 4+1+1+8*16+8+8+4+8*2048*2 = 32922, rounded up
#define DEFAULT_CHUNK_SIZE ((uint64_t)1 << 30)  // 1GB

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

static int write_header(FILE* f, file_header_t* hdr) {
    size_t n = fwrite(hdr, 1, HEADER_SIZE, f);
    return (n == HEADER_SIZE) ? 0 : -1;
}

#define FAIL(fmt, ...)                        \
    do {                                      \
        if (f) fclose(f);                     \
        if (fname) mxFree(fname);             \
        mexErrMsgIdAndTxt(fmt, __VA_ARGS__);  \
    } while (0)

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt("save_lz4_mex:NumArgs", "Usage: save_lz4_mex(filename, array)");

    char* fname = mxArrayToString(prhs[0]);
    if (!fname || strlen(fname) == 0)
        FAIL("save_lz4_mex:BadFilename", "Could not extract a valid filename.");

    FILE* f = fopen(fname, "wb");
    if (!f)
        FAIL("save_lz4_mex:OpenFailed", "Failed to open file '%s': %s", fname, strerror(errno));

    const mxArray* arr = prhs[1];
    mwSize ndims = mxGetNumberOfDimensions(arr);
    const mwSize* dims = mxGetDimensions(arr);

    // --- Prepare header ---
    file_header_t hdr;
    memset(&hdr, 0, sizeof(file_header_t));
    hdr.magic = MAGIC_NUMBER;
    hdr.ndims = (uint8_t)ndims;

    if (mxIsDouble(arr)) hdr.dtype = DT_DOUBLE;
    else if (mxIsSingle(arr)) hdr.dtype = DT_SINGLE;
    else if (mxIsUint16(arr)) hdr.dtype = DT_UINT16;
    else
        FAIL("save_lz4_mex:BadType", "Only double, single, uint16 are supported.");

    size_t el_sz = mxGetElementSize(arr);
    uint64_t numel = (uint64_t)mxGetNumberOfElements(arr);
    hdr.total_uncompressed = el_sz * numel;

    for (mwSize i = 0; i < ndims && i < MAX_DIMS; ++i)
        hdr.dims[i] = (uint64_t)dims[i];

    hdr.chunk_size = DEFAULT_CHUNK_SIZE;
    uint32_t n_chunks = (uint32_t)((hdr.total_uncompressed + hdr.chunk_size - 1) / hdr.chunk_size);
    if (n_chunks > MAX_CHUNKS)
        FAIL("save_lz4_mex:TooManyChunks", "Too many chunks. Increase MAX_CHUNKS or chunk size.");

    hdr.num_chunks = n_chunks;

    if (fseek(f, 0, SEEK_SET) != 0 || write_header(f, &hdr) != 0)
        FAIL("save_lz4_mex:HeaderWriteFailed", "Failed to write placeholder header.");

    const char* src = (const char*)mxGetData(arr);
    uint64_t offset = 0;

    for (uint32_t i = 0; i < n_chunks; ++i)
    {
        uint64_t this_uncomp = hdr.chunk_size;
        if (offset + this_uncomp > hdr.total_uncompressed)
            this_uncomp = hdr.total_uncompressed - offset;
        hdr.chunk_uncomp[i] = this_uncomp;

        int uncomp_size = (int)this_uncomp;
        int max_dst = LZ4_compressBound(uncomp_size);
        if (max_dst <= 0)
            FAIL("save_lz4_mex:LZ4BoundError", "LZ4_compressBound returned invalid size.");

        char* cbuf = (char*)mxMalloc(max_dst);
        if (!cbuf)
            FAIL("save_lz4_mex:AllocFailed", "Out of memory.");

        int comp_bytes = LZ4_compress_default(src + offset, cbuf, uncomp_size, max_dst);
        if (comp_bytes <= 0) {
            mxFree(cbuf);
            FAIL("save_lz4_mex:LZ4CompressFail", "LZ4 compression failed.");
        }

        hdr.chunk_comp[i] = (uint64_t)comp_bytes;

        size_t written = fwrite(cbuf, 1, comp_bytes, f);
        mxFree(cbuf);
        if (written != (size_t)comp_bytes)
            FAIL("save_lz4_mex:WriteFail", "Failed to write compressed chunk.");

        offset += this_uncomp;
    }

    if (fflush(f) != 0 || fseek(f, 0, SEEK_SET) != 0 || write_header(f, &hdr) != 0)
        FAIL("save_lz4_mex:FinalHeaderWriteFail", "Failed to write final header.");

    if (fflush(f) != 0 || fclose(f) != 0)
        FAIL("save_lz4_mex:FileCloseFail", "Could not close file properly.");

    mxFree(fname);
}
