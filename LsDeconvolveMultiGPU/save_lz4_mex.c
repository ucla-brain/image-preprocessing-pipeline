// save_lz4_mex.c (Chunked LZ4 for >2GB arrays)
// Requires lz4.c and lz4.h in the project folder
#include "mex.h"
#include "lz4.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h> // for unlink

// --- CONSTANTS ---
#define MAX_DIMS 16
#define MAX_CHUNKS 2048
#define HEADER_SIZE 33280  // >= 4+1+1+8*16+8+8+4+8*2048*2 = 32922, rounded up
#define DEFAULT_CHUNK_SIZE ((uint64_t)1<<30) // 1GB

#ifdef _WIN32
#include <io.h>
#define unlink _unlink
#endif

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
    // Padding to ensure header is always HEADER_SIZE bytes
    uint8_t padding[HEADER_SIZE - (4+1+1+8*MAX_DIMS+8+8+4+8*MAX_CHUNKS*2)];
} file_header_t;

void write_header(FILE* f, file_header_t* hdr) {
    size_t n = fwrite(hdr, 1, HEADER_SIZE, f);
    if (n != (size_t)HEADER_SIZE) mexErrMsgIdAndTxt("save_lz4_mex:write_header", "Failed to write data to file");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2) mexErrMsgIdAndTxt("save_lz4_mex:NumArgs", "Usage: save_lz4_mex(filename, array)");

    // === Robustly extract filename from char or string ===
    char fname[4096] = {0};
    char* tempstr = mxArrayToString(prhs[0]);
    if (tempstr && strlen(tempstr) < sizeof(fname)) {
        strncpy(fname, tempstr, sizeof(fname)-1);
        fname[sizeof(fname)-1] = '\0';
        mxFree(tempstr);
    } else {
        if (tempstr) mxFree(tempstr);
        mexErrMsgIdAndTxt("save_lz4_mex:BadString", "Could not extract filename. Make sure to pass a char array (e.g., 'file.lz4') or a scalar string (e.g., \"file.lz4\")");
    }

    if (strlen(fname) == 0) {
        mexErrMsgIdAndTxt("save_lz4_mex:EmptyFilename", "Filename is empty!");
    }

    mexPrintf("Trying to open: %s\n", fname);
    unlink(fname); // Try to remove any existing file, ignore errors

    FILE* f = fopen(fname, "wb");
    if (!f) {
        mexPrintf("fopen failed: errno = %d (%s)\n", errno, strerror(errno));
        mexErrMsgIdAndTxt("save_lz4_mex:OpenFailed", "Failed to open file for writing: %s (errno %d: %s)", fname, errno, strerror(errno));
    }

    const mxArray* arr = prhs[1];
    mwSize ndims = mxGetNumberOfDimensions(arr);
    const mwSize* dims = mxGetDimensions(arr);

    // --- Prepare header ---
    file_header_t hdr = {0};
    hdr.magic = MAGIC_NUMBER;
    hdr.ndims = (uint8_t)ndims;

    // Dtype
    if (mxIsDouble(arr)) hdr.dtype = DT_DOUBLE;
    else if (mxIsSingle(arr)) hdr.dtype = DT_SINGLE;
    else if (mxIsUint16(arr)) hdr.dtype = DT_UINT16;
    else {
        fclose(f);
        mexErrMsgTxt("Only double, single, uint16 are supported.");
    }

    size_t el_sz = mxGetElementSize(arr);
    uint64_t numel = (uint64_t)mxGetNumberOfElements(arr);
    hdr.total_uncompressed = (uint64_t)el_sz * numel;
    for (mwSize i=0; i<ndims && i<MAX_DIMS; ++i)
        hdr.dims[i] = (uint64_t)dims[i];

    // Chunk logic
    hdr.chunk_size = DEFAULT_CHUNK_SIZE;
    uint32_t n_chunks = (uint32_t)((hdr.total_uncompressed + hdr.chunk_size - 1) / hdr.chunk_size);
    hdr.num_chunks = n_chunks;
    if (n_chunks > MAX_CHUNKS) {
        fclose(f);
        mexErrMsgTxt("Too many chunks. Increase MAX_CHUNKS or chunk size.");
    }

    // --- Write placeholder header (will overwrite after chunk sizes known) ---
    fseek(f, 0, SEEK_SET);
    write_header(f, &hdr);

    const char* src = (const char*)mxGetData(arr);
    uint64_t offset = 0;

    for (uint32_t i=0; i<n_chunks; ++i)
    {
        uint64_t this_uncomp = hdr.chunk_size;
        if (offset + this_uncomp > hdr.total_uncompressed)
            this_uncomp = hdr.total_uncompressed - offset;
        hdr.chunk_uncomp[i] = this_uncomp;

        int uncomp_size = (int)this_uncomp; // always <= 1GB (checked by logic above)
        int max_dst = LZ4_compressBound(uncomp_size);

        char* cbuf = (char*)mxMalloc(max_dst);

        int comp_bytes = LZ4_compress_default(src + offset, cbuf, uncomp_size, max_dst);
        if (comp_bytes <= 0) {
            fclose(f); mxFree(cbuf);
            mexErrMsgTxt("LZ4 compression failed for chunk.");
        }
        hdr.chunk_comp[i] = (uint64_t)comp_bytes;

        // Write compressed chunk
        size_t written = fwrite(cbuf, 1, comp_bytes, f);
        if (written != (size_t)comp_bytes) {
            fclose(f); mxFree(cbuf);
            mexErrMsgTxt("Failed to write compressed chunk data.");
        }

        mxFree(cbuf);
        offset += this_uncomp;
    }

    // --- Write real header with actual chunk sizes ---
    fflush(f);
    fseek(f, 0, SEEK_SET);
    write_header(f, &hdr);
    fflush(f);
    if (fclose(f) != 0) mexErrMsgIdAndTxt("save_lz4_mex:FileCloseFailed", "Could not close file (write error?)");
}
