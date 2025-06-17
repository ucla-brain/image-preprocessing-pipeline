/*==============================================================================
  save_bl_tif.cpp
  -----------------------------------------------------------------------------
  High-throughput Z-slice saver for 3-D MATLAB arrays (1 TIFF per Z-slice)

  Author:       Keivan Moradi (in collaboration with ChatGPT-4o)
  License:      GNU General Public License v3.0 (https://www.gnu.org/licenses/)

  OVERVIEW
  -------
  • Purpose:
      Efficiently saves each Z-slice from a 3D MATLAB array to a separate TIFF
      file using LZW, Deflate, or no compression. Supports [X Y Z] or [Y X Z] input.

  • Highlights:
      – Accepts `uint8` or `uint16` MATLAB input arrays.
      – Fully cross-platform (uses libtiff).
      – Supports multithreading.
      – Compression: none, lzw, or deflate.
      – Matches `load_bl_tif.cpp` slice order and dimensions.

  PARALLELISM
  -----------
  • Parallel I/O is implemented using atomic index dispatching:
      Each worker thread atomically claims the next available task index
      using `std::atomic<size_t>::fetch_add`, avoiding locks or queues.
      This model scales efficiently for uniform workloads like per-slice saves.

  USAGE
  -----
      save_bl_tif(array3d, fileList, orderFlag, compression)

      • array3d      : 3D numeric array, uint8 or uint16
      • fileList     : 1xZ cell array of full path strings
      • orderFlag    : (logical or uint32 scalar)
                         true  = [X Y Z] input
                         false = [Y X Z] input (MATLAB default)
      • compression  : string: "none", "lzw", or "deflate"

  ==============================================================================*/

#include "mex.h"
#include "matrix.h"
#include "tiffio.h"
#include <atomic>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// -----------------------------------------------------------------------------
struct MatlabString {
    char* p;
    explicit MatlabString(const mxArray* a) : p(mxArrayToUTF8String(a)) {
        if (!p) mexErrMsgIdAndTxt("save_bl_tif:BadString",
                                  "Failed to convert string");
    }
    ~MatlabString() { mxFree(p); }
    const char* get() const { return p; }
    MatlabString(const MatlabString&) = delete;
    MatlabString& operator=(const MatlabString&) = delete;
};

struct SaveTask {
    const uint8_t* base;
    mwSize dim0, dim1;          // rows (Y), cols (X) in MATLAB memory
    mwSize z;
    std::string path;
    bool isXYZ;
    mxClassID cls;
    std::string comp;
};

// -----------------------------------------------------------------------------
void save_slice(const SaveTask& t)
{
    const size_t es = (t.cls == mxUINT16_CLASS) ? 2 : 1;
    const mwSize W  = t.isXYZ ? t.dim0 : t.dim1;   // after transpose?
    const mwSize H  = t.isXYZ ? t.dim1 : t.dim0;
    const size_t sliceOff = static_cast<size_t>(t.z) * t.dim0 * t.dim1;

    TIFF* tif = TIFFOpen(t.path.c_str(), "w");
    if (!tif) throw std::runtime_error("Cannot open " + t.path);

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,  W);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, H);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, es == 2 ? 16 : 8);
    uint16_t c = (t.comp == "lzw") ? COMPRESSION_LZW :
                 (t.comp == "deflate") ? COMPRESSION_DEFLATE :
                 COMPRESSION_NONE;
    TIFFSetField(tif, TIFFTAG_COMPRESSION, c);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, H);

    std::vector<uint8_t> scan(W * es);

    for (mwSize y = 0; y < H; ++y) {
        for (mwSize x = 0; x < W; ++x) {
            size_t srcIdx = t.isXYZ
                ?  x + y * t.dim0 + sliceOff    // transpose: (y,x) -> (x,y)
                :  y + x * t.dim0 + sliceOff;   // native order
            std::memcpy(&scan[x * es], t.base + srcIdx * es, es);
        }
        if (TIFFWriteScanline(tif, scan.data(), y, 0) < 0) {
            TIFFClose(tif);
            throw std::runtime_error("TIFFWriteScanline failed at row "
                                     + std::to_string(y));
        }
    }
    TIFFClose(tif);
}

// -----------------------------------------------------------------------------
void worker(const std::vector<SaveTask>& T,
            std::atomic_size_t& next,
            std::vector<std::string>& errs,
            std::mutex& mtx)
{
    size_t i;
    while ((i = next.fetch_add(1)) < T.size()) {
        try { save_slice(T[i]); }
        catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(mtx);
            errs.emplace_back(e.what());
        }
    }
}

// -----------------------------------------------------------------------------
void mexFunction(int, mxArray*[], int nr, const mxArray* pr[])
{
    try {
        if (nr != 4)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                "Usage: save_bl_tif(vol, fileList, orderFlag, compression)");

        const mxArray* V = pr[0];
        if (!mxIsUint8(V) && !mxIsUint16(V))
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                "Volume must be uint8 or uint16");
        if (mxGetNumberOfDimensions(V) != 3)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be 3-D");

        mwSize dim0 = mxGetDimensions(V)[0];  // rows Y
        mwSize dim1 = mxGetDimensions(V)[1];  // cols X
        mwSize dim2 = mxGetDimensions(V)[2];  // slices Z

        if (!mxIsCell(pr[1]) || mxGetNumberOfElements(pr[1]) != dim2)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                "fileList must be 1xZ cell array");

        bool isXYZ;
        if (mxIsLogicalScalar(pr[2])) isXYZ = mxIsLogicalScalarTrue(pr[2]);
        else if (mxIsUint32(pr[2]) || mxIsInt32(pr[2]))
            isXYZ = (*static_cast<uint32_t*>(mxGetData(pr[2])) != 0);
        else  mexErrMsgIdAndTxt("save_bl_tif:Input",
                "orderFlag must be logical or int32/uint32 scalar");

        MatlabString cs(pr[3]);
        std::string comp(cs.get());
        if (comp!="none"&&comp!="lzw"&&comp!="deflate")
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "compression must be none|lzw|deflate");

        const uint8_t* base = static_cast<const uint8_t*>(mxGetData(V));

        std::vector<SaveTask> tasks; tasks.reserve(dim2);
        for (mwSize z=0; z<dim2; ++z) {
            const mxArray* c = mxGetCell(pr[1],z);
            if (!mxIsChar(c))
                mexErrMsgIdAndTxt("save_bl_tif:Input",
                                  "fileList entries must be char");
            MatlabString p(c);
            tasks.push_back({base, dim0, dim1, z, p.get(),
                             isXYZ, mxGetClassID(V), comp});
        }

        std::atomic_size_t next(0);
        std::vector<std::string> errs;
        std::mutex mtx;
        unsigned nT = std::max(1u,
           std::min<unsigned>(std::thread::hardware_concurrency(), dim2));
        std::vector<std::thread> pool;
        for (unsigned t=0;t<nT;++t)
            pool.emplace_back(worker,std::cref(tasks),std::ref(next),
                              std::ref(errs),std::ref(mtx));
        for (auto& th:pool) th.join();

        if (!errs.empty()) {
            std::string msg="save_bl_tif errors:\n";
            for (auto&s:errs) msg+="  - "+s+"\n";
            mexErrMsgIdAndTxt("save_bl_tif:Runtime","%s",msg.c_str());
        }
    }
    catch (const std::exception& e) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception","%s",e.what());
    }
}
