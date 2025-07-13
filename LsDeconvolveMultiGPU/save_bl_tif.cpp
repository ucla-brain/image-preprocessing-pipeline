/*==============================================================================
  save_bl_tif.cpp

  Ultra-high-throughput, NUMA- and socket-aware, multi-threaded TIFF Z-slice saver
  for 3D MATLAB volumes. Uses explicit async producer-consumer I/O, advanced file
  flags, and hwloc-driven thread pairing for maximum locality and throughput.

  USAGE:
    save_bl_tif(volume, fileList, isXYZ, compression[, nThreads, useTiles]);

  INPUTS:
    • volume      : 3D MATLAB array (uint8 or uint16), or 2D for single slice.
    • fileList    : 1×Z cell array of output filenames, one per Z-slice.
    • isXYZ       : Scalar logical/numeric. True if 'volume' is [X Y Z], false
                    if [Y X Z].
    • compression : String. "none", "lzw", or "deflate".
    • nThreads    : (Optional) Number of thread-pairs. Default = half of
                    available cores. Pass [] to auto-select.
    • useTiles    : (Optional) true for tiled TIFF, false for strip mode.

  HIGHLIGHTS:
    • **Async producer-consumer** with lock-limited bounded queue sized at
      2×threadCount for smooth pipelining.
    • **NUMA- and socket-aware affinity** via hwloc: each producer/consumer pair
      is bound to sibling PUs on the same core or node.
    • **Direct I/O flags**: FILE_FLAG_WRITE_THROUGH on Windows replaces
      double FlushFileBuffers; buffered I/O on Linux (no O_DIRECT) avoids
      misaligned-syscall overhead.
    • **UTF-8 → UTF-16** on Windows via MultiByteToWideChar (no deprecated
      <codecvt> usage).
    • **Robust atomic output**: write to “.tmp”, flush/close, then rename with
      copy-delete fallback for maximum atomicity.
    • **Guarded strip/tile sizes**: default rowsPerStrip = 1; all dimensions
      clamped to ≥1.
    • **Macro-safe std::min/max**: calls wrapped as (std::min)/(std::max) to
      avoid Windows min/max macro collisions.
    • **Early-abort on error**: atomic<bool> flag stops all threads on first
      exception.
    • **RAII + C++17 best practices**: no resource leaks, clear ownership.
    • **Extensible tuning**: compile-time constants for tile edge, queue
      capacity, etc.
    • **Build integration**: prebuilt hwloc on Windows; autotools-driven hwloc
      on Linux.

  DEPENDENCIES:
    • libtiff ≥ 4.7, hwloc ≥ 2.12 (prebuilt on Windows, autotools on Linux)
    • MATLAB MEX API, C++17, <filesystem>, POSIX/Windows threading, hwloc.h.

  EXAMPLES:
    % Async strip-mode LZW on [X Y Z] volume with default threads
    save_bl_tif(vol, fileList, true, 'lzw');
    % Async tile-mode deflate with 8 pairs
    save_bl_tif(vol, fileList, true, 'deflate', 8, true);

  AUTHOR:
    Keivan Moradi (with ChatGPT assistance), 2025

  LICENSE:
    GNU GPL v3 — https://www.gnu.org/licenses/gpl-3.0.html
==============================================================================*/

#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING

#include "mex.h"
#include "tiffio.h"
#include "mex_thread_utils.hpp"

#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <stdexcept>
#include <system_error>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>
#include <optional>
#include <cassert>
#include <bitset>
#include <set>
#include <map>
#include <immintrin.h>
#include <cstddef>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
    #undef min
    #undef max
    #include <io.h>
    #include <fcntl.h>
    #ifndef W_OK
        #define W_OK 2
    #endif
    #define access _access
#elif defined(__linux__) || defined(__APPLE__)
    #include <sched.h>
    #include <pthread.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <libgen.h>
#endif

namespace fs = std::filesystem;

// Compile-time parameters for TIFF
static constexpr uint32_t kRowsPerStripDefault = 1;
static constexpr uint32_t kOptimalTileEdge     = 128;
// How many logical producer-consumer pairs to group together per queue (set to 1 for 1:1 mapping)
static constexpr size_t kWires = 1;
// Threshold for large copy (tunable)
static constexpr size_t kNonTemporalCopyThreshold = 128 * 1024;

// Choose threshold and alignment width based on the *widest* supported instruction set:
#if defined(__AVX512F__)
    static constexpr size_t kAlignment = 64;
#elif defined(__AVX2__) || defined(__AVX__)
    static constexpr size_t kAlignment = 32;
#elif defined(__SSE2__)
    static constexpr size_t kAlignment = 16;
#else
    static constexpr size_t kAlignment = 1;
#endif

namespace platform {
#if defined(_WIN32)
    inline std::wstring utf8_to_utf16(const std::string& utf8)
    {
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(),
                                              static_cast<int>(utf8.size()), nullptr, 0);
        std::wstring wstr(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), static_cast<int>(utf8.size()),
                            &wstr[0], size_needed);
        return wstr;
    }
    inline fs::path toPlatformPath(const std::string& utf8) {
        return fs::path(utf8);
    }
#else
    inline const std::string& toPlatformPath(const std::string& utf8) { return utf8; }
#endif
}

/**
 * @brief Pick a good tile size for TIFF tiling.
 */
inline void pick_tile_size(uint32_t width, uint32_t height,
                           uint32_t& tileWidth, uint32_t& tileLength) {
    if (width >= kOptimalTileEdge && height >= kOptimalTileEdge) {
        tileWidth  = kOptimalTileEdge;
        tileLength = kOptimalTileEdge;
    } else {
        tileWidth = tileLength = 64;
    }
}

//=========================
//    SliceWriteTask
//=========================
/**
 * @brief Structure representing all info needed to write a single TIFF slice.
 */
struct SliceWriteTask {
    void*                sliceBuffer;
    uint32_t             sliceIndex;
    uint32_t             widthPixels;
    uint32_t             heightPixels;
    uint32_t             bytesPerPixel;
    std::string          outputFilePath;
    bool                 isXYZLayout;
    uint16_t             compressionTag;
    bool                 useTiles;
};

//=========================
//    TemporaryFileGuard
//=========================
/**
 * @brief RAII guard for a temporary file.
 */
class TemporaryFileGuard {
public:
    TemporaryFileGuard(const fs::path& path) : tempPath_(path) {}
    ~TemporaryFileGuard() {
        if (fs::exists(tempPath_)) {
            std::error_code ec;
            fs::remove(tempPath_, ec);
        }
    }
    const fs::path& get() const { return tempPath_; }
private:
    fs::path tempPath_;
};

//=========================
//   TiffWriterDirect
//=========================
/**
 * @brief RAII-safe cross-platform TIFF writer.
 */
struct TiffWriterDirect
{
    TIFF*       tiffHandle = nullptr;
    std::string filePath;
#if defined(__linux__) || defined(__APPLE__)
    int fileDescriptor = -1;
#endif

    explicit TiffWriterDirect(const std::string& filePath_)
        : filePath(filePath_)
    {
#if defined(_WIN32)
        std::wstring wPath = platform::utf8_to_utf16(filePath_);
        constexpr size_t kLimit = 248;
        if (wPath.size() >= kLimit && wPath.rfind(LR"(\\?\)", 0) == std::wstring::npos)
            wPath.insert(0, LR"(\\?\)");
        tiffHandle = TIFFOpenW(wPath.c_str(), "w");
        if (!tiffHandle)
            throw std::runtime_error("TIFFOpenW failed for: " + filePath_);
#elif defined(__linux__) || defined(__APPLE__)
        fileDescriptor = ::open(filePath_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
        if (fileDescriptor == -1)
            throw std::runtime_error("open() failed for: " + filePath_ + " errno=" + std::to_string(errno));
        tiffHandle = TIFFFdOpen(fileDescriptor, filePath_.c_str(), "w");
        if (!tiffHandle) {
            ::close(fileDescriptor);
            throw std::runtime_error("TIFFFdOpen failed for: " + filePath_);
        }
#else
        tiffHandle = TIFFOpen(filePath_.c_str(), "w");
        if (!tiffHandle)
            throw std::runtime_error("TIFFOpen failed for: " + filePath_);
#endif
    }
    ~TiffWriterDirect() noexcept
    {
        if (!tiffHandle) return;
#if defined(__linux__) || defined(__APPLE__)
        ::fsync(TIFFFileno(tiffHandle));
        std::vector<char> dirbuf(filePath.c_str(), filePath.c_str() + filePath.size() + 1);
        char* dir = dirname(dirbuf.data());
        int dirfd = ::open(dir, O_RDONLY);
        if (dirfd != -1) { ::fsync(dirfd); ::close(dirfd); }
#endif
        TIFFClose(tiffHandle);
        tiffHandle = nullptr;
    }
    TiffWriterDirect(const TiffWriterDirect&)            = delete;
    TiffWriterDirect& operator=(const TiffWriterDirect&) = delete;
};

//=========================
//      File Helpers
//=========================
inline void make_writable(const fs::path& path) {
#if defined(_WIN32)
    SetFileAttributesW(platform::utf8_to_utf16(path.string()).c_str(), FILE_ATTRIBUTE_NORMAL);
#else
    ::chmod(path.c_str(), 0666);
#endif
}

/**
 * @brief Copy src to dst, then delete src. Used as fallback if rename fails.
 */
inline bool copy_and_delete(const fs::path& src, const fs::path& dst, std::string& errorMessage) {
    std::error_code ec;
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        errorMessage = "copy_file failed: " + ec.message();
        return false;
    }
    fs::remove(src, ec);
    if (ec) {
        errorMessage = "remove(src) after copy failed: " + ec.message();
    }
    return true;
}

/**
 * @brief Atomically move/replace file; fallback to copy-delete if needed.
 */
inline bool robust_move_or_replace(const fs::path& src,
                                   const fs::path& dst,
                                   std::string&    errorMessage) {
    make_writable(src);
    make_writable(dst);
#if defined(_WIN32)
    std::wstring wideSrc = platform::utf8_to_utf16(src.string());
    std::wstring wideDst = platform::utf8_to_utf16(dst.string());
    for (int attempt = 0; attempt < 5; ++attempt) {
        if (ReplaceFileW(wideDst.c_str(), wideSrc.c_str(),
                         nullptr, REPLACEFILE_WRITE_THROUGH, nullptr, nullptr))
            return true;
        DWORD err = GetLastError();
        if (err == ERROR_SHARING_VIOLATION || err == ERROR_ACCESS_DENIED) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10 * (attempt + 1)));
            continue;
        }
        errorMessage = "ReplaceFileW error: " + std::to_string(err);
        break;
    }
    if (MoveFileExW(wideSrc.c_str(), wideDst.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        return true;
    errorMessage = "MoveFileExW failed: " + std::to_string(GetLastError());
#elif defined(__linux__) || defined(__APPLE__)
    std::error_code ec;
    for (int attempt = 0; attempt < 5; ++attempt) {
        fs::rename(src, dst, ec);
        if (!ec) return true;
        if (fs::exists(dst)) fs::remove(dst, ec);
        std::this_thread::sleep_for(std::chrono::milliseconds(10 * (attempt + 1)));
    }
    errorMessage = "rename failed: " + ec.message();
#else
    std::error_code ec;
    fs::rename(src, dst, ec);
    if (!ec) return true;
    errorMessage = "rename failed";
#endif
    std::string copyDeleteMsg;
    if (copy_and_delete(src, dst, copyDeleteMsg))
        return true;
    errorMessage += " | copy-delete fallback: " + copyDeleteMsg;
    return false;
}

//=========================
//  Slice TIFF Writer
//=========================
static void write_slice_to_tiff(const SliceWriteTask& task)
{
    const fs::path tempPath = fs::path(task.outputFilePath).concat(".tmp");
    TemporaryFileGuard tempGuard(tempPath);

    // Write TIFF into a temp file (closed before rename)
    {
        TiffWriterDirect tiffWriter(tempPath.string());
        TIFF* tiffHandle = tiffWriter.tiffHandle;

        TIFFSetField(tiffHandle, TIFFTAG_IMAGEWIDTH,  (std::max)(1u, task.widthPixels));
        TIFFSetField(tiffHandle, TIFFTAG_IMAGELENGTH, (std::max)(1u, task.heightPixels));
        TIFFSetField(tiffHandle, TIFFTAG_BITSPERSAMPLE, static_cast<uint16_t>(task.bytesPerPixel * 8));
        TIFFSetField(tiffHandle, TIFFTAG_SAMPLESPERPIXEL, static_cast<uint16_t>(1));
        TIFFSetField(tiffHandle, TIFFTAG_COMPRESSION, task.compressionTag);
        TIFFSetField(tiffHandle, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tiffHandle, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

        if (task.compressionTag == COMPRESSION_ADOBE_DEFLATE) {
            TIFFSetField(tiffHandle, TIFFTAG_ZIPQUALITY, 1);
            TIFFSetField(tiffHandle, TIFFTAG_PREDICTOR,  1);
        }

        if (task.useTiles) {
            uint32_t tileW, tileH;
            pick_tile_size(task.widthPixels, task.heightPixels, tileW, tileH);
            tileW = (std::max)(1u, tileW);
            tileH = (std::max)(1u, tileH);
            TIFFSetField(tiffHandle, TIFFTAG_TILEWIDTH,  tileW);
            TIFFSetField(tiffHandle, TIFFTAG_TILELENGTH, tileH);

            const uint32_t tilesX = (task.widthPixels  + tileW - 1) / tileW;
            const uint32_t tilesY = (task.heightPixels + tileH - 1) / tileH;
            const size_t   tileBytes = size_t(tileW) * tileH * task.bytesPerPixel;

            if (task.isXYZLayout) {
                std::vector<uint8_t> tile(tileBytes);
                for (uint32_t ty = 0; ty < tilesY; ++ty)
                for (uint32_t tx = 0; tx < tilesX; ++tx) {
                    const uint32_t y0   = ty * tileH;
                    const uint32_t rows = (std::min)(tileH, task.heightPixels - y0);
                    const uint32_t x0   = tx * tileW;
                    const uint32_t cols = (std::min)(tileW, task.widthPixels  - x0);

                    std::fill(tile.begin(), tile.end(), 0);

                    for (uint32_t r = 0; r < rows; ++r)
                        std::memcpy(&tile[(r * tileW) * task.bytesPerPixel],
                                    static_cast<uint8_t*>(task.sliceBuffer) +
                                        ((size_t(y0 + r) * task.widthPixels + x0) *
                                        task.bytesPerPixel),
                                    size_t(cols) * task.bytesPerPixel);

                    tstrip_t tileIdx = TIFFComputeTile(tiffHandle, x0, y0, 0, 0);
                    if (TIFFWriteEncodedTile(tiffHandle, tileIdx,
                                             tile.data(), tileBytes) < 0)
                        throw std::runtime_error("Tile write failed");
                }
            } else {
                thread_local std::vector<uint8_t> tile;
                if (tile.size() < tileBytes) tile.resize(tileBytes);
                for (uint32_t ty = 0; ty < tilesY; ++ty)
                for (uint32_t tx = 0; tx < tilesX; ++tx) {
                    const uint32_t y0 = ty * tileH;
                    const uint32_t rows = (std::min)(tileH, task.heightPixels - y0);
                    const uint32_t x0 = tx * tileW;
                    const uint32_t cols = (std::min)(tileW, task.widthPixels - x0);

                    std::fill(tile.begin(), tile.begin() + rows*tileW*task.bytesPerPixel, 0);

                    for (uint32_t row = 0; row < rows; ++row) {
                        const uint8_t* src = static_cast<uint8_t*>(task.sliceBuffer) +
                            ((size_t(y0 + row) + size_t(x0) * task.heightPixels) *
                              task.bytesPerPixel);
                        uint8_t* dst = tile.data() + row * tileW * task.bytesPerPixel;
                        for (uint32_t col = 0; col < cols; ++col) {
                            std::memcpy(dst + col * task.bytesPerPixel,
                                        src + col * task.heightPixels * task.bytesPerPixel,
                                        task.bytesPerPixel);
                        }
                    }
                    tstrip_t tileIdx = TIFFComputeTile(tiffHandle, x0, y0, 0, 0);
                    if (TIFFWriteEncodedTile(tiffHandle, tileIdx,
                                             tile.data(), tileBytes) < 0)
                        throw std::runtime_error("Tile write failed");
                }
            }
        } else {
            // Use strip mode with minimum rowsPerStrip = 1
            const uint32_t rowsPerStrip =
                (std::max)(1u, (std::min)(kRowsPerStripDefault, task.heightPixels));
            TIFFSetField(tiffHandle, TIFFTAG_ROWSPERSTRIP, rowsPerStrip);
            const uint32_t totalStrips =
                (task.heightPixels + rowsPerStrip - 1) / rowsPerStrip;

            if (task.isXYZLayout) {
                for (uint32_t strip = 0; strip < totalStrips; ++strip) {
                    const uint32_t y0   = strip * rowsPerStrip;
                    const uint32_t rows = (std::min)(rowsPerStrip, task.heightPixels - y0);
                    const size_t bytesThisStrip =
                        size_t(task.widthPixels) * rows * task.bytesPerPixel;
                    const uint8_t* src =
                        static_cast<uint8_t*>(task.sliceBuffer) +
                        size_t(y0) * task.widthPixels * task.bytesPerPixel;

                    if (TIFFWriteEncodedStrip(tiffHandle, strip,
                                              const_cast<uint8_t*>(src),
                                              bytesThisStrip) < 0)
                        throw std::runtime_error("Strip write failed");
                }
            } else {
                thread_local std::vector<uint8_t> stripBuffer;
                const size_t maxStripBytes =
                    size_t(task.widthPixels) * rowsPerStrip * task.bytesPerPixel;
                if (stripBuffer.size() < maxStripBytes)
                    stripBuffer.resize(maxStripBytes);

                for (uint32_t strip = 0; strip < totalStrips; ++strip) {
                    const uint32_t y0   = strip * rowsPerStrip;
                    const uint32_t rows = (std::min)(rowsPerStrip, task.heightPixels - y0);
                    for (uint32_t row = 0; row < rows; ++row) {
                        const uint8_t* src = static_cast<uint8_t*>(task.sliceBuffer) +
                            ((size_t(y0 + row)) * task.bytesPerPixel);
                        uint8_t* dst = stripBuffer.data() +
                            row * task.widthPixels * task.bytesPerPixel;
                        for (uint32_t col = 0; col < task.widthPixels; ++col) {
                            std::memcpy(dst + col * task.bytesPerPixel,
                                        src + col * task.heightPixels * task.bytesPerPixel,
                                        task.bytesPerPixel);
                        }
                    }
                    const size_t bytesThisStrip =
                        size_t(task.widthPixels) * rows * task.bytesPerPixel;
                    if (TIFFWriteEncodedStrip(tiffHandle, strip,
                                              stripBuffer.data(),
                                              bytesThisStrip) < 0)
                        throw std::runtime_error("Strip write failed");
                }
            }
        }

        TIFFFlush(tiffHandle);
    }
    if (!fs::exists(tempPath))
        throw std::runtime_error("Temp file vanished: " + tempPath.string());

    std::string moveError;
    if (!robust_move_or_replace(tempPath, task.outputFilePath, moveError))
        throw std::runtime_error("Atomic move failed: " + moveError);
}

inline bool is_aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

inline void fast_slice_copy(void* dst, const void* src, size_t bytes) {
#if defined(__AVX512F__)
    if (bytes >= kNonTemporalCopyThreshold && is_aligned(src, 64) && is_aligned(dst, 64) && (bytes % 64 == 0)) {
        auto* d = reinterpret_cast<__m512i*>(dst);
        const auto* s = reinterpret_cast<const __m512i*>(src);
        size_t n = bytes / 64;
        for (size_t i = 0; i < n; ++i) {
            _mm512_stream_si512(d + i, _mm512_load_si512(s + i));
        }
        _mm_sfence();
        return;
    }
#elif defined(__AVX2__) || defined(__AVX__)
    if (bytes >= kNonTemporalCopyThreshold && is_aligned(src, 32) && is_aligned(dst, 32) && (bytes % 32 == 0)) {
        auto* d = reinterpret_cast<__m256i*>(dst);
        const auto* s = reinterpret_cast<const __m256i*>(src);
        size_t n = bytes / 32;
        for (size_t i = 0; i < n; ++i) {
            _mm256_stream_si256(d + i, _mm256_load_si256(s + i));
        }
        _mm_sfence();
        return;
    }
#elif defined(__SSE2__)
    if (bytes >= kNonTemporalCopyThreshold && is_aligned(src, 16) && is_aligned(dst, 16) && (bytes % 16 == 0)) {
        auto* d = reinterpret_cast<__m128i*>(dst);
        const auto* s = reinterpret_cast<const __m128i*>(src);
        size_t n = bytes / 16;
        for (size_t i = 0; i < n; ++i) {
            _mm_stream_si128(d + i, _mm_load_si128(s + i));
        }
        _mm_sfence();
        return;
    }
#endif
    std::memcpy(dst, src, bytes);
}

//=========================
//        mexFunction
//=========================
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {

    ensure_hwloc_initialized();
    try {
        if (nrhs < 4 || nrhs > 6)
            mexErrMsgIdAndTxt("save_bl_tif:usage",
                "Usage: save_bl_tif(volume, fileList, isXYZ, compression[, nThreads, useTiles]);");

        if (!mxIsUint8(prhs[0]) && !mxIsUint16(prhs[0]))
            mexErrMsgIdAndTxt("save_bl_tif:type",
                              "Volume must be uint8 or uint16.");

        const mwSize* rawDims   = mxGetDimensions(prhs[0]);
        const uint32_t rawRows  = static_cast<uint32_t>(rawDims[0]);
        const uint32_t rawCols  = static_cast<uint32_t>(rawDims[1]);
        const uint32_t numSlices =
            (mxGetNumberOfDimensions(prhs[0]) == 3 ?
             static_cast<uint32_t>(rawDims[2]) : 1);

        const bool isXYZ =
            mxIsLogicalScalarTrue(prhs[2]) ||
            (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0);

        const uint32_t widthPixels  = isXYZ ? rawRows : rawCols;
        const uint32_t heightPixels = isXYZ ? rawCols : rawRows;

        char* compCStr = mxArrayToUTF8String(prhs[3]);
        std::string compressionStr(compCStr);
        mxFree(compCStr);

        uint16_t compressionTag =
              compressionStr == "none"    ? COMPRESSION_NONE
            : compressionStr == "lzw"     ? COMPRESSION_LZW
            : compressionStr == "deflate" ? COMPRESSION_ADOBE_DEFLATE
            : throw std::runtime_error("Invalid compression: " + compressionStr);

        if (!mxIsCell(prhs[1]) ||
            mxGetNumberOfElements(prhs[1]) != numSlices)
            mexErrMsgIdAndTxt("save_bl_tif:files",
                              "fileList must be a cell array with one entry per slice.");

        std::vector<std::string> outputPaths(numSlices);
        for (uint32_t i = 0; i < numSlices; ++i) {
            char* pathChar = mxArrayToUTF8String(mxGetCell(prhs[1], i));
            outputPaths[i] = pathChar;
            mxFree(pathChar);
        }

        for (const auto& path : outputPaths) {
            fs::path dir = fs::path(path).parent_path();
            if (!dir.empty() && !fs::exists(dir))
                mexErrMsgIdAndTxt("save_bl_tif:invalidPath",
                    "Directory does not exist: %s", dir.string().c_str());
            if (fs::exists(path) && access(path.c_str(), W_OK) != 0)
                mexErrMsgIdAndTxt("save_bl_tif:readonly",
                    "Cannot overwrite read-only file: %s", path.c_str());
        }

        const uint8_t* volumePtr =
            static_cast<const uint8_t*>(mxGetData(prhs[0]));
        const uint32_t bytesPerPixel =
            (mxGetClassID(prhs[0]) == mxUINT16_CLASS ? 2u : 1u);

        const size_t totalLogicalCores = get_available_cores();
        const size_t defaultThreads = std::max(totalLogicalCores / 2, size_t(1));
        const size_t requestedThreads =
            (nrhs >= 5 && !mxIsEmpty(prhs[4]))
                ? static_cast<size_t>(mxGetScalar(prhs[4]))
                : defaultThreads;
        const size_t threadPairCount =
            std::min(requestedThreads, static_cast<size_t>(numSlices));

        const bool useTiles =
            (nrhs >= 6) ? (mxIsLogicalScalarTrue(prhs[5]) ||
                           (mxIsNumeric(prhs[5]) && mxGetScalar(prhs[5]) != 0))
                        : false;

        // ==========================================================================
        //      Main producer-consumer queue tuning: set to 1 for maximal locality
        // ==========================================================================
        static_assert(kWires >= 1, "kWires must be at least 1.");
        const size_t numWires = threadPairCount / kWires + ((threadPairCount % kWires) ? 1 : 0);

        // Vector of per-wire (per-pair) queues. Each wire is a producer-consumer pair sharing a dedicated queue.
        std::vector<std::unique_ptr<BoundedQueue<std::shared_ptr<SliceWriteTask>>>> queuesForWires;
        queuesForWires.reserve(numWires);
        for (size_t w = 0; w < numWires; ++w)
            queuesForWires.emplace_back(std::make_unique<BoundedQueue<std::shared_ptr<SliceWriteTask>>>(2 * kWires)); // Bounded to 2×pair size for pipelining

        std::vector<std::thread> producerThreads, consumerThreads;
        producerThreads.reserve(threadPairCount);
        consumerThreads.reserve(threadPairCount);

        std::atomic<uint32_t> nextSliceIndex{0};
        std::vector<std::string> runtimeErrors;
        std::mutex              errorMutex;
        std::atomic<bool>       abortFlag{false};
        auto threadPairs = assign_thread_affinity_pairs(threadPairCount);
        const size_t sliceBytes = size_t(widthPixels) * heightPixels * bytesPerPixel;

        // ---- Launch threads: each pair uses its own queue ("wire") ----
        for (size_t t = 0; t < threadPairCount; ++t) {
            // Each pair t uses queuesForWires[t / kWires]
            BoundedQueue<std::shared_ptr<SliceWriteTask>>& queueForPair = *queuesForWires[t / kWires];

            // Producer: allocates memory (NUMA-local), prepares slice, pushes to queue
            producerThreads.emplace_back([&, t] {
                set_thread_affinity(threadPairs[t].producerLogicalCore);
                unsigned numaNode = threadPairs[t].numaNode;

                while (true) {
                    if (abortFlag.load(std::memory_order_acquire)) break;
                    uint32_t idx = nextSliceIndex.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= numSlices) break;


                    auto task = std::make_shared<SliceWriteTask>();
                    task->sliceBuffer    = allocate_numa_local_buffer(g_hwlocTopo->get(), sliceBytes, numaNode);
                    assert(is_aligned(task->sliceBuffer, kAlignment));
                    //std::memcpy(task->sliceBuffer, volumePtr + idx * sliceBytes, sliceBytes);
                    fast_slice_copy(task->sliceBuffer, volumePtr + idx * sliceBytes, sliceBytes);
                    task->sliceIndex     = idx;
                    task->widthPixels    = widthPixels;
                    task->heightPixels   = heightPixels;
                    task->bytesPerPixel  = bytesPerPixel;
                    task->outputFilePath = outputPaths[idx];
                    task->isXYZLayout    = isXYZ;
                    task->compressionTag = compressionTag;
                    task->useTiles       = useTiles;
                    queueForPair.push(std::move(task));
                }
                // Signal end-of-tasks to consumer(s)
                queueForPair.push(nullptr);
            });

            // Consumer: pops tasks from queue, writes to disk, aborts on error
            consumerThreads.emplace_back([&, t] {
                set_thread_affinity(threadPairs[t].consumerLogicalCore);
                while (true) {
                    if (abortFlag.load(std::memory_order_acquire)) break;
                    std::shared_ptr<SliceWriteTask> task;
                    queueForPair.wait_and_pop(task);
                    if (!task) break; // End-of-tasks signal
                    try {
                        write_slice_to_tiff(*task);
                    } catch (const std::exception& ex) {
                        free_numa_local_buffer(g_hwlocTopo->get(), task->sliceBuffer, sliceBytes);
                        abortFlag.store(true, std::memory_order_release);
                        std::lock_guard<std::mutex> lock(errorMutex);
                        runtimeErrors.emplace_back(ex.what());
                        break;
                    }
                    free_numa_local_buffer(g_hwlocTopo->get(), task->sliceBuffer, sliceBytes);
                }
            });
        }

        // Wait for all producers and consumers to finish
        for (auto& p : producerThreads) p.join();
        for (auto& c : consumerThreads) c.join();

        if (!runtimeErrors.empty())
            mexErrMsgIdAndTxt("save_bl_tif:runtime", runtimeErrors.front().c_str());

        if (nlhs > 0)
            plhs[0] = const_cast<mxArray*>(prhs[0]);
    } catch (const std::exception& ex) {
        mexErrMsgIdAndTxt("save_bl_tif:runtime", ex.what());
    }
}
