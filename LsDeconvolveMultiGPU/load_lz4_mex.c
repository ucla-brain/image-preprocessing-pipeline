// load_lz4_mex.c (Chunked LZ4 for >2GB arrays)
// Requires lz4.c and lz4.h in the project folder
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

enum dtype_enum { DT_DOUBLE=1, DT_SINGLE=2, DT_UINT16=3 };

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
    uint8_t padding[HEADER_SIZE - (4+1+1+8*MAX_DIMS+8+8+4+8*MAX_CHUNKS*2)];
} file_header_t;

void read_header(FILE* f, file_header_t* hdr) {
    size_t got = fread(hdr, 1, HEADER_SIZE, f);
    if (got != HEADER_SIZE)
        mexErrMsgTxt("Failed to read file header.");
    if (hdr->magic != MAGIC_NUMBER)
        mexErrMsgTxt("Invalid file: magic number mismatch.");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 1)
        mexErrMsgTxt("Usage: arr = load_lz4_mex(filename)");

    char fname[1024];
    mxGetString(prhs[0], fname, sizeof(fname));
    FILE* f = fopen(fname, "rb");
    if (!f) mexErrMsgTxt("Failed to open file for reading.");

    file_header_t hdr = {0};
    read_header(f, &hdr);

    // Get dimensions
    mwSize ndims = hdr.ndims;
    mwSize dims[MAX_DIMS];
    for (mwSize i=0; i<ndims && i<MAX_DIMS; ++i)
        dims[i] = (mwSize)hdr.dims[i];

    // Allocate output MATLAB array
    mxClassID classid;
    if (hdr.dtype == DT_DOUBLE) classid = mxDOUBLE_CLASS;
    else if (hdr.dtype == DT_SINGLE) classid = mxSINGLE_CLASS;
    else if (hdr.dtype == DT_UINT16) classid = mxUINT16_CLASS;
    else {
        fclose(f);
        mexErrMsgTxt("Unsupported dtype in file.");
    }

    mxArray* arr = mxCreateNumericArray(ndims, dims, classid, mxREAL);
    char* dest = (char*)mxGetData(arr);
    uint64_t total_uncompressed = hdr.total_uncompressed;
    uint64_t copied = 0;

    // Read and decompress each chunk
    for (uint32_t i=0; i<hdr.num_chunks; ++i)
    {
        uint64_t this_uncomp = hdr.chunk_uncomp[i];
        uint64_t this_comp = hdr.chunk_comp[i];
        if (this_uncomp > 0x7FFFFFFF || this_comp > 0x7FFFFFFF) {
            fclose(f);
            mexErrMsgTxt("LZ4: Single chunk too large. File corrupt or incompatible.");
        }

        int uncomp_size = (int)this_uncomp;
        int comp_size = (int)this_comp;

        char* cbuf = (char*)mxMalloc(comp_size);
        size_t nread = fread(cbuf, 1, comp_size, f);
        if (nread != (size_t)comp_size) {
            fclose(f); mxFree(cbuf);
            mexErrMsgTxt("Failed to read compressed chunk.");
        }

        int dsize = LZ4_decompress_safe(cbuf, dest + copied, comp_size, uncomp_size);
        mxFree(cbuf);
        if (dsize < 0 || dsize != uncomp_size) {
            fclose(f);
            mexErrMsgTxt("LZ4 decompression failed or file corrupt.");
        }

        copied += this_uncomp;
    }

    fclose(f);
    plhs[0] = arr;
}
