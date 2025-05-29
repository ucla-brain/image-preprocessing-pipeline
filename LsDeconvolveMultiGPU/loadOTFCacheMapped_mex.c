#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define META_LINE_LENGTH 256

// Helper: read shape from .meta file (expects a line: "shape [x y z]")
int read_shape(const char* metafile, mwSize shape[3]) {
    FILE* f = fopen(metafile, "r");
    if (!f) return 1;
    char line[META_LINE_LENGTH];
    while (fgets(line, META_LINE_LENGTH, f)) {
        char* shape_ptr = strstr(line, "shape");
        if (shape_ptr) {
            int n = sscanf(line, "shape [%zu %zu %zu]", &shape[0], &shape[1], &shape[2]);
            if (n == 3) {
                fclose(f);
                return 0; // Success
            }
        }
    }
    fclose(f);
    return 2; // Shape not found
}

// Main MEX function
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // Check arguments
    if (nrhs != 1 || !mxIsChar(prhs[0])) {
        mexErrMsgTxt("Usage: [otf, otf_conj] = loadOTFCacheMapped_mex(filename_base)");
    }

    // Get base filename
    char filename_base[1024];
    mxGetString(prhs[0], filename_base, sizeof(filename_base));

    // Build .meta filename
    char meta_fname[1100];
    snprintf(meta_fname, sizeof(meta_fname), "%s.meta", filename_base);

    // Read shape
    mwSize shape[3];
    int shape_status = read_shape(meta_fname, shape);
    if (shape_status == 1)
        mexErrMsgIdAndTxt("loadOTFCacheMapped_mex:MetaOpen", "Cannot open .meta file: %s", meta_fname);
    if (shape_status == 2)
        mexErrMsgIdAndTxt("loadOTFCacheMapped_mex:MetaParse", "Could not parse shape from: %s", meta_fname);

    mwSize n_elems = shape[0] * shape[1] * shape[2];

    // Build .bin filename
    char bin_fname[1100];
    snprintf(bin_fname, sizeof(bin_fname), "%s.bin", filename_base);

    FILE* f = fopen(bin_fname, "rb");
    if (!f)
        mexErrMsgIdAndTxt("loadOTFCacheMapped_mex:BinOpen", "Cannot open .bin file: %s", bin_fname);

    // Allocate MATLAB outputs: [shape] complex single, both outputs
    plhs[0] = mxCreateNumericArray(3, shape, mxSINGLE_CLASS, mxCOMPLEX); // otf
    plhs[1] = mxCreateNumericArray(3, shape, mxSINGLE_CLASS, mxCOMPLEX); // otf_conj

    float* otf_real = (float*) mxGetData(plhs[0]);
    float* otf_imag = (float*) mxGetImagData(plhs[0]);
    float* conj_real = (float*) mxGetData(plhs[1]);
    float* conj_imag = (float*) mxGetImagData(plhs[1]);

    size_t total = 4 * n_elems;
    float* buf = (float*) mxMalloc(total * sizeof(float));

    size_t nread = fread(buf, sizeof(float), total, f);
    fclose(f);

    if (nread != total) {
        mxFree(buf);
        mexErrMsgIdAndTxt("loadOTFCacheMapped_mex:BinRead", "Read error: expected %zu floats, got %zu", total, nread);
    }

    // Fill OTF outputs
    for (mwSize i = 0; i < n_elems; ++i) {
        otf_real[i]  = buf[i];
        otf_imag[i]  = buf[i + n_elems];
        conj_real[i] = buf[i + 2 * n_elems];
        conj_imag[i] = buf[i + 3 * n_elems];
    }
    mxFree(buf);
}
