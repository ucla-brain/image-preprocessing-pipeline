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
        - Header is 33 KiB and written twice:
              once as a placeholder before compression,
              and again at the end with final chunk sizes.
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
#define HEADER_SIZE 33280  // 33 KiB, enough for large arrays
#define DEFAULT_CHUNK_SIZE ((uint64_t)1 << 30)  // 1 GiB

#define MAGIC_NUMBER 0x4C5A4331  // 'LZ4C'

enum dtype_enum { DT_DOUBLE = 1, DT_SINGLE = 2, DT_UINT16 = 3 };

typedef struct {
    uint32_t magic;
    uint8_t  dtype;
    uint8_t  ndims;
    uint64_t dims[MAX_DIMS];
    uint64_t total_uncompressed;
    uint64_t chunk_size;
    uint32_t num_chunks;
    uint64_t chunk_uncomp[MAX_CHUNKS];
    uint64_t chunk_comp[MAX_CHUNKS];
    uint8_t  padding[HEADER_SIZE - (4 + 1 + 1 + 8 * MAX_DIMS + 8 + 8 + 4 + 8 * MAX_CHUNKS * 2)];
} file_header_t;

// --- Safe write of header block ---
static int write_header(FILE* file, const file_header_t* header) {
    return (fwrite(header, 1, HEADER_SIZE, file) == HEADER_SIZE) ? 0 : -1;
}

// --- Cleanup-and-error macro ---
#define FAIL(id, ...)                        \
    do {                                     \
        if (file) fclose(file);              \
        if (filename) mxFree(filename);      \
        mexErrMsgIdAndTxt(id, __VA_ARGS__);  \
    } while (0)

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)nlhs; (void)plhs;  // unused

    if (nrhs != 2)
        mexErrMsgIdAndTxt("save_lz4_mex:NumArgs", "Usage: save_lz4_mex(filename, array)");

    char* filename = mxArrayToString(prhs[0]);
    FILE* file = NULL;

    if (!filename || strlen(filename) == 0)
        FAIL("save_lz4_mex:BadFilename", "Filename must be a non-empty string or char array.");

    file = fopen(filename, "wb");
    if (!file)
        FAIL("save_lz4_mex:OpenFailed", "Failed to open file '%s': %s", filename, strerror(errno));

    const mxArray* input_array = prhs[1];
    mwSize num_dims = mxGetNumberOfDimensions(input_array);
    const mwSize* dims = mxGetDimensions(input_array);

    // --- Validate supported types ---
    file_header_t header;
    memset(&header, 0, sizeof(header));
    header.magic = MAGIC_NUMBER;
    header.ndims = (uint8_t)num_dims;

    if (mxIsDouble(input_array))       header.dtype = DT_DOUBLE;
    else if (mxIsSingle(input_array))  header.dtype = DT_SINGLE;
    else if (mxIsUint16(input_array))  header.dtype = DT_UINT16;
    else
        FAIL("save_lz4_mex:BadType", "Only double, single, and uint16 arrays are supported.");

    uint64_t element_size = (uint64_t)mxGetElementSize(input_array);
    uint64_t total_elements = (uint64_t)mxGetNumberOfElements(input_array);
    header.total_uncompressed = element_size * total_elements;

    for (mwSize i = 0; i < num_dims && i < MAX_DIMS; ++i)
        header.dims[i] = (uint64_t)dims[i];

    // --- Chunk planning ---
    header.chunk_size = DEFAULT_CHUNK_SIZE;
    uint32_t num_chunks = (uint32_t)((header.total_uncompressed + header.chunk_size - 1) / header.chunk_size);
    if (num_chunks > MAX_CHUNKS)
        FAIL("save_lz4_mex:TooManyChunks", "Too many chunks. Increase MAX_CHUNKS or chunk size.");

    header.num_chunks = num_chunks;

    // --- Write placeholder header ---
    if (fseek(file, 0, SEEK_SET) != 0 || write_header(file, &header) != 0)
        FAIL("save_lz4_mex:HeaderWriteFailed", "Could not write header.");

    // --- Compress and write each chunk ---
    const char* src_data = (const char*)mxGetData(input_array);
    if (!src_data)
        FAIL("save_lz4_mex:NoData", "Input array has no data.");

    uint64_t offset_bytes = 0;

    for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
    {
        uint64_t this_chunk_bytes = header.chunk_size;
        if (offset_bytes + this_chunk_bytes > header.total_uncompressed)
            this_chunk_bytes = header.total_uncompressed - offset_bytes;

        header.chunk_uncomp[chunk_idx] = this_chunk_bytes;

        int uncomp_size = (int)this_chunk_bytes;
        int max_compressed_size = LZ4_compressBound(uncomp_size);
        if (max_compressed_size <= 0)
            FAIL("save_lz4_mex:LZ4BoundError", "LZ4_compressBound failed.");

        char* compressed_buf = (char*)mxMalloc(max_compressed_size);
        if (!compressed_buf)
            FAIL("save_lz4_mex:AllocFailed", "Out of memory.");

        int compressed_bytes = LZ4_compress_default(src_data + offset_bytes, compressed_buf, uncomp_size, max_compressed_size);
        if (compressed_bytes <= 0) {
            mxFree(compressed_buf);
            FAIL("save_lz4_mex:CompressFail", "LZ4 compression failed.");
        }

        header.chunk_comp[chunk_idx] = (uint64_t)compressed_bytes;

        size_t bytes_written = fwrite(compressed_buf, 1, compressed_bytes, file);
        mxFree(compressed_buf);
        if (bytes_written != (size_t)compressed_bytes)
            FAIL("save_lz4_mex:WriteFail", "Failed to write compressed data.");

        offset_bytes += this_chunk_bytes;
    }

    // --- Rewrite final header with actual chunk sizes ---
    if (fflush(file) != 0 || fseek(file, 0, SEEK_SET) != 0 || write_header(file, &header) != 0)
        FAIL("save_lz4_mex:FinalHeaderWriteFail", "Could not update header.");

    if (fflush(file) != 0 || fclose(file) != 0)
        FAIL("save_lz4_mex:CloseFailed", "Could not close file.");

    mxFree(filename);
}
