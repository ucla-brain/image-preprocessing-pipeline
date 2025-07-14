/*==============================================================================
  load_slab_lz4_save_as_tif.cpp
================================================================================*/
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING

#include "mex.h"
#include "tiffio.h"
#include "mex_thread_utils.hpp"
#include "lz4.h"

#include <cstdio>
#include <cmath>
#include <chrono>
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


/*------------------------------ fmaf fallback --------------------------------*/
#ifndef fmaf
#  define fmaf(a,b,c) ((a)*(b)+(c))
#endif

/*========================== Thread-local Scratch Buffers =====================*/
template<typename T>
struct ThreadScratch {
    static thread_local std::vector<T> uncompressed;
    static thread_local std::vector<char> compressed;
    ~ThreadScratch() { uncompressed.clear(); compressed.clear(); }
};
template<typename T> thread_local std::vector<T> ThreadScratch<T>::uncompressed;
template<typename T> thread_local std::vector<char> ThreadScratch<T>::compressed;

/*=========================== Brick Header Parsing ============================*/
constexpr uint32_t MAGIC = 0x4C5A4331U;  // 'LZC1'
constexpr uint32_t HDR_BYTES = 33280;
constexpr uint32_t MAX_CHUNKS = 2048;
enum DType : uint8_t { DT_DOUBLE = 1, DT_SINGLE = 2, DT_UINT16 = 3 };

struct BrickHeader {
    uint32_t magic;
    uint8_t dtype, ndims;
    uint64_t dims[16];
    uint64_t totalBytes, chunkBytes;
    uint32_t nChunks;
    uint64_t uLen[MAX_CHUNKS], cLen[MAX_CHUNKS];
    uint8_t _pad[HDR_BYTES - (4 + 1 + 1 + 16*8 + 8 + 8 + 4 + MAX_CHUNKS*16)];
};

static void freadExact(FILE* fp, void* dst, std::size_t n, const char* ctx) {
    if (std::fread(dst, 1, n, fp) != n)
        throw std::runtime_error(std::string(ctx) + ": I/O error");
}

static BrickHeader readHeader(FILE* fp, const std::string& file) {
    BrickHeader h{};
    freadExact(fp, &h, HDR_BYTES, "header");
    if (h.magic != MAGIC) throw std::runtime_error(file + ": bad magic");
    if (h.dtype != DT_SINGLE) throw std::runtime_error(file + ": not float32");
    if (h.nChunks == 0 || h.nChunks > MAX_CHUNKS)
        throw std::runtime_error(file + ": bad chunk count");
    return h;
}

inline uint64_t idx3d(uint64_t x, uint64_t y, uint64_t z,
                      uint64_t dimX, uint64_t dimY) {
    return x + dimX * (y + dimY * z);
}

/*============================= Brick Job Struct ==============================*/
template<typename OUT_T>
struct BrickJob {
    std::string file;
    uint64_t x0, y0, z0, x1, y1, z1, dimX, dimY, dimZ;
    OUT_T* dst;
    float scal, ampl, dmin, dmax;

    void operator()() const {
        auto& bufferFloat = ThreadScratch<float>::uncompressed;
        auto& bufferCompressed = ThreadScratch<char>::compressed;

        std::unique_ptr<FILE, decltype(&std::fclose)> fp(std::fopen(file.c_str(), "rb"), &std::fclose);
        if (!fp) throw std::runtime_error("open " + file);

        const auto header = readHeader(fp.get(), file);

        uint64_t bx = x1 - x0 + 1, by = y1 - y0 + 1, bz = z1 - z0 + 1;
        uint64_t voxelCount = bx * by * bz;
        if (bufferFloat.size() < voxelCount) bufferFloat.resize(voxelCount);

        char* decompressed = reinterpret_cast<char*>(bufferFloat.data());
        uint64_t offset = 0;

        for (uint32_t i = 0; i < header.nChunks; ++i) {
            if (bufferCompressed.size() < header.cLen[i])
                bufferCompressed.resize(header.cLen[i]);
            freadExact(fp.get(), bufferCompressed.data(), header.cLen[i], "chunk");
            int written = LZ4_decompress_safe(bufferCompressed.data(), decompressed + offset,
                                              int(header.cLen[i]), int(header.uLen[i]));
            if (written < 0 || uint64_t(written) != header.uLen[i])
                throw std::runtime_error(file + ": LZ4 error");
            offset += header.uLen[i];
        }

        if (offset != header.totalBytes)
            throw std::runtime_error(file + ": size mismatch");

        const float kLinear = scal * ampl / dmax;
        const float kMinMax = (dmin > 0.f) ? scal * ampl / (dmax - dmin) : kLinear;
        const bool useMinMax = (dmin > 0.f);

        const float* src = bufferFloat.data();
        for (uint64_t z = 0; z < bz; ++z)
        for (uint64_t y = 0; y < by; ++y) {
            const uint64_t base = idx3d(x0, y0 + y, z0 + z, dimX, dimY);
            for (uint64_t x = 0; x < bx; ++x) {
                float val = src[(z * by + y) * bx + x];
                if (useMinMax)
                    val = (val - dmin) * kMinMax;
                else
                    val = val * kLinear;

                val -= ampl;
                val = (val >= 0.f) ? std::floor(val + 0.5f) : std::ceil(val - 0.5f);
                val = std::clamp(val, 0.f, scal);
                dst[base + x] = static_cast<OUT_T>(val);
            }
        }
    }
};

/*============================ Atomic Thread Pool =============================*/
template<typename JobT>
void run_atomic_thread_pool(const std::vector<JobT>& jobs, int nThreads) {
    std::atomic<size_t> jobIndex{0};
    std::atomic<bool> hasException{false};
    std::string errorMsg;

    auto worker = [&]() {
        while (true) {
            size_t idx = jobIndex.fetch_add(1, std::memory_order_relaxed);
            if (idx >= jobs.size() || hasException.load()) break;
            try {
                jobs[idx]();
            } catch (const std::exception& e) {
                if (!hasException.exchange(true))
                    errorMsg = e.what();
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(nThreads);
    for (int t = 0; t < nThreads; ++t)
        threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    if (hasException)
        throw std::runtime_error(errorMsg);
}

//======================================================================================================================

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
    void*                sliceBuffer;      // Pointer to the slice pixels
    bool                 ownsBuffer;       // True ⇢ producer must free after consumer
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
//        loadSlabLz4
//=========================

struct ValidatedInputs {
    std::vector<std::string> srcFiles;
    std::vector<std::string> dstFiles;
    std::vector<uint64_t> dims;  // [dimX, dimY, dimZ]
    float scal, ampl, dmin, dmax;
    std::string compression;
    bool useTiles;
    int maxThreads;
    mxClassID outType;
    const mxArray *p1, *p2;
    size_t nBricks;
    size_t nSlices;
    bool isXYZLayout;   // <-- ADD THIS LINE
};

// Template function: assembles per-slice output buffers from LZ4 bricks.
// Each output buffer is width*height, suitable for direct TIFF writing.
template<typename OUT_T>
std::vector<std::unique_ptr<OUT_T[]>> loadSlabLz4(const ValidatedInputs& inp) {
    const size_t nBricks = inp.nBricks;
    const size_t nSlices = inp.nSlices;
    const uint64_t dimX = inp.dims[0], dimY = inp.dims[1], dimZ = inp.dims[2];
    const size_t sliceSize = dimX * dimY;

    // Atomic array of pointers, one per output slice (all start as nullptr)
    std::vector<std::atomic<OUT_T*>> slices(nSlices);
    for (size_t i = 0; i < nSlices; ++i)
        slices[i].store(nullptr, std::memory_order_relaxed);

    struct Job {
        std::string file;
        uint64_t x0, y0, z0, x1, y1, z1;
        float scal, ampl, dmin, dmax;
    };
    std::vector<Job> jobs;
    jobs.reserve(nBricks);

    auto getCoord = [](const mxArray* arr, mwSize idx) -> uint64_t {
        return mxIsUint64(arr)
            ? reinterpret_cast<const uint64_t*>(mxGetData(arr))[idx]
            : static_cast<uint64_t>(mxGetPr(arr)[idx]);
    };

    // Prepare brick jobs
    for (size_t i = 0; i < nBricks; ++i) {
        jobs.push_back(Job{
            inp.srcFiles[i],
            getCoord(inp.p1, i)-1, getCoord(inp.p1, i+nBricks)-1, getCoord(inp.p1, i+2*nBricks)-1,
            getCoord(inp.p2, i)-1, getCoord(inp.p2, i+nBricks)-1, getCoord(inp.p2, i+2*nBricks)-1,
            inp.scal, inp.ampl, inp.dmin, inp.dmax
        });
    }

    std::atomic<size_t> jobIndex{0};
    std::exception_ptr errorPtr = nullptr;

    auto worker = [&]() {
        while (true) {
            size_t idx = jobIndex.fetch_add(1, std::memory_order_relaxed);
            if (idx >= jobs.size() || errorPtr) break;
            try {
                const Job& job = jobs[idx];
                auto& bufferFloat = ThreadScratch<float>::uncompressed;
                auto& bufferCompressed = ThreadScratch<char>::compressed;

                std::unique_ptr<FILE, decltype(&std::fclose)> fp(std::fopen(job.file.c_str(), "rb"), &std::fclose);
                if (!fp) throw std::runtime_error("open " + job.file);

                const auto header = readHeader(fp.get(), job.file);

                uint64_t bx = job.x1 - job.x0 + 1;
                uint64_t by = job.y1 - job.y0 + 1;
                uint64_t bz = job.z1 - job.z0 + 1;
                uint64_t brickVoxels = bx * by * bz;
                if (bufferFloat.size() < brickVoxels) bufferFloat.resize(brickVoxels);

                char* decompressed = reinterpret_cast<char*>(bufferFloat.data());
                uint64_t offset = 0;
                for (uint32_t i = 0; i < header.nChunks; ++i) {
                    if (bufferCompressed.size() < header.cLen[i])
                        bufferCompressed.resize(header.cLen[i]);
                    freadExact(fp.get(), bufferCompressed.data(), header.cLen[i], "chunk");
                    int written = LZ4_decompress_safe(bufferCompressed.data(), decompressed + offset,
                                                      int(header.cLen[i]), int(header.uLen[i]));
                    if (written < 0 || uint64_t(written) != header.uLen[i])
                        throw std::runtime_error(job.file + ": LZ4 error");
                    offset += header.uLen[i];
                }
                if (offset != header.totalBytes)
                    throw std::runtime_error(job.file + ": size mismatch");

                const float kLinear = job.scal * job.ampl / job.dmax;
                const float kMinMax = (job.dmin > 0.f) ? job.scal * job.ampl / (job.dmax - job.dmin) : kLinear;
                const bool useMinMax = (job.dmin > 0.f);

                // For each Z-plane in this brick, copy/correct region into global output buffer
                for (uint64_t bzIdx = 0; bzIdx < bz; ++bzIdx) {
                    uint64_t globalZ = job.z0 + bzIdx;
                    if (globalZ >= inp.nSlices) continue; // skip out-of-bounds

                    // Allocate buffer only if this slice hasn't been allocated
                    OUT_T* outSlice = slices[globalZ].load(std::memory_order_acquire);
                    if (!outSlice) {
                        // Use unique_ptr for exception safety, always freed if not released
                        std::unique_ptr<OUT_T[]> newBuf(new OUT_T[sliceSize]);
                        OUT_T* expected = nullptr;
                        // Only install if it hasn't been set (thread-safe)
                        if (slices[globalZ].compare_exchange_strong(expected, newBuf.get(),
                                std::memory_order_acq_rel, std::memory_order_acquire)) {
                            std::fill_n(newBuf.get(), sliceSize, OUT_T{});
                            outSlice = newBuf.release(); // we "won" – transfer ownership to atomic array
                        } else {
                            outSlice = expected; // another thread beat us, unique_ptr auto-deletes
                        }
                    }
                    // Copy bx-by region from brick into correct region of output slice
                    for (uint64_t byIdx = 0; byIdx < by; ++byIdx) {
                        uint64_t globalY = job.y0 + byIdx;
                        if (globalY >= dimY) continue;
                        for (uint64_t bxIdx = 0; bxIdx < bx; ++bxIdx) {
                            uint64_t globalX = job.x0 + bxIdx;
                            if (globalX >= dimX) continue;
                            size_t slabIdx  = bxIdx + bx * (byIdx + by * bzIdx);      // [bx, by, bz]
                            size_t outIdx   = globalX + dimX * globalY;                // [x + X*y]
                            float val = bufferFloat[slabIdx];
                            if (useMinMax)
                                val = (val - job.dmin) * kMinMax;
                            else
                                val = val * kLinear;
                            val -= job.ampl;
                            val = (val >= 0.f) ? std::floor(val + 0.5f) : std::ceil(val - 0.5f);
                            val = std::clamp(val, 0.f, job.scal);
                            outSlice[outIdx] = static_cast<OUT_T>(val);
                        }
                    }
                }
            } catch (...) {
                if (!errorPtr) errorPtr = std::current_exception();
            }
        }
    };

    // Launch threads
    size_t nThreads = std::min(inp.maxThreads > 0 ? inp.maxThreads : get_available_cores(), jobs.size());
    std::vector<std::thread> threads;
    threads.reserve(nThreads);
    for (size_t t = 0; t < nThreads; ++t)
        threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    if (errorPtr) std::rethrow_exception(errorPtr);

    // Transfer pointers to unique_ptr for safe ownership and automatic cleanup
    std::vector<std::unique_ptr<OUT_T[]>> outSlices(nSlices);
    for (size_t i = 0; i < nSlices; ++i)
        outSlices[i].reset(slices[i].exchange(nullptr));

    return outSlices;
}

//=========================
//        saveSlabTif
//=========================

template<typename T>
void saveSlabTif(const std::vector<T*>& slices, const ValidatedInputs& inp)
{
    const size_t numSlices = inp.nSlices;
    const size_t widthPixels  = inp.dims[0];
    const size_t heightPixels = inp.dims[1];
    const size_t bytesPerPixel = (inp.outType == mxUINT16_CLASS ? 2 : 1);

    // Map compression string to TIFF constant
    uint16_t compressionTag =
          inp.compression == "none"    ? COMPRESSION_NONE
        : inp.compression == "lzw"     ? COMPRESSION_LZW
        : inp.compression == "deflate" ? COMPRESSION_ADOBE_DEFLATE
        : throw std::runtime_error("Invalid compression: " + inp.compression);

    // Threading setup
    const size_t totalLogicalCores = get_available_cores();
    const size_t threadPairCount   = std::min<size_t>(inp.maxThreads, numSlices);

    static_assert(kWires >= 1, "kWires must be at least 1.");
    const size_t numWires = threadPairCount / kWires + ((threadPairCount % kWires) ? 1 : 0);

    std::vector<std::unique_ptr<BoundedQueue<std::shared_ptr<SliceWriteTask>>>> queuesForWires;
    queuesForWires.reserve(numWires);
    for (size_t w = 0; w < numWires; ++w)
        queuesForWires.emplace_back(
            std::make_unique<BoundedQueue<std::shared_ptr<SliceWriteTask>>>(2 * kWires));

    std::vector<std::thread> producerThreads, consumerThreads;
    producerThreads.reserve(threadPairCount);
    consumerThreads.reserve(threadPairCount);

    std::atomic<uint32_t> nextSliceIndex{0};
    std::vector<std::string> runtimeErrors;
    std::mutex              errorMutex;
    std::atomic<bool>       abortFlag{false};
    auto threadPairs = assign_thread_affinity_pairs(threadPairCount);
    const size_t sliceBytes = widthPixels * heightPixels * bytesPerPixel;

    // PRODUCERS: Each producer determines buffer ownership per-slice
    for (size_t t = 0; t < threadPairCount; ++t) {
        auto& queueForPair = *queuesForWires[t / kWires];
        producerThreads.emplace_back([&, t] {
            set_thread_affinity(threadPairs[t].producerLogicalCore);
            const unsigned numaNode = threadPairs[t].numaNode;

            while (true) {
                if (abortFlag.load(std::memory_order_acquire)) break;

                const uint32_t idx = nextSliceIndex.fetch_add(1, std::memory_order_relaxed);
                if (idx >= numSlices) break;

                auto task               = std::make_shared<SliceWriteTask>();
                task->sliceIndex        = idx;
                task->widthPixels       = widthPixels;
                task->heightPixels      = heightPixels;
                task->bytesPerPixel     = bytesPerPixel;
                task->outputFilePath    = inp.dstFiles[idx];
                task->isXYZLayout       = inp.isXYZLayout;
                task->compressionTag    = compressionTag;
                task->useTiles          = inp.useTiles;

                const bool canAlias = (inp.isXYZLayout && !inp.useTiles);

                if (canAlias) {
                    task->sliceBuffer = const_cast<T*>(slices[idx]);
                    task->ownsBuffer  = false;
                    warm_pages_async(task->sliceBuffer, sliceBytes);
                } else {
                    task->sliceBuffer = allocate_numa_local_buffer(g_hwlocTopo->get(), sliceBytes, numaNode);
                    if (!task->sliceBuffer)
                        throw std::runtime_error("NUMA allocation failed");
                    fast_slice_copy(task->sliceBuffer, slices[idx], sliceBytes);
                    task->ownsBuffer  = true;
                }

                queueForPair.push(std::move(task));
            }
            queueForPair.push(nullptr); // poison pill
        });

        // CONSUMERS: Free buffer only if we allocated it
        consumerThreads.emplace_back([&, t] {
            set_thread_affinity(threadPairs[t].consumerLogicalCore);
            while (true) {
                if (abortFlag.load(std::memory_order_acquire)) break;
                std::shared_ptr<SliceWriteTask> task;
                queueForPair.wait_and_pop(task);
                if (!task) break;

                try {
                    write_slice_to_tiff(*task);
                } catch (const std::exception& ex) {
                    if (task->ownsBuffer)
                        free_numa_local_buffer(g_hwlocTopo->get(), task->sliceBuffer, sliceBytes);
                    abortFlag.store(true, std::memory_order_release);
                    std::lock_guard<std::mutex> lock(errorMutex);
                    runtimeErrors.emplace_back(ex.what());
                    break;
                }

                if (task->ownsBuffer)
                    free_numa_local_buffer(g_hwlocTopo->get(), task->sliceBuffer, sliceBytes);
            }
        });
    }

    // Wait for all threads
    for (auto& p : producerThreads)  p.join();
    for (auto& c : consumerThreads)  c.join();

    if (!runtimeErrors.empty())
        mexErrMsgIdAndTxt("load_slab_lz4_save_as_tif:runtime", runtimeErrors.front().c_str());
}

ValidatedInputs validate_inputs(const mxArray* prhs[], int nrhs)
{
    // === Minimum number of required args ===
    if (nrhs < 9)
        mexErrMsgIdAndTxt("load_slab_lz4_save_as_tif:usage",
            "Usage: load_slab_lz4_save_as_tif(srcFiles, dstFiles, p1, p2, dims, scal, ampl, dmin, dmax [, compression, useTiles, maxThreads, isXYZLayout])");

    // srcFiles and dstFiles
    if (!mxIsCell(prhs[0]) || !mxIsCell(prhs[1]))
        mexErrMsgIdAndTxt("load_slab_lz4_save_as_tif:cell",
            "srcFiles and dstFiles must be cell arrays.");
    size_t nBricks = mxGetNumberOfElements(prhs[0]);
    size_t nSlices = mxGetNumberOfElements(prhs[1]);
    if (nBricks == 0 || nSlices == 0)
        mexErrMsgIdAndTxt("load_slab_lz4_save_as_tif:dim",
            "srcFiles and dstFiles must be non-empty cell arrays.");

    // dims (should be 3)
    if (mxGetNumberOfElements(prhs[4]) != 3)
        mexErrMsgIdAndTxt("load_slab_lz4_save_as_tif:dims",
            "dims must be a vector of 3 elements [dimX, dimY, dimZ].");
    std::vector<uint64_t> dims(3);
    for (int i = 0; i < 3; ++i)
        dims[i] = mxIsUint64(prhs[4])
            ? reinterpret_cast<const uint64_t*>(mxGetData(prhs[4]))[i]
            : static_cast<uint64_t>(mxGetPr(prhs[4])[i]);

    // Numeric scalars
    #define GRAB_FLOAT(idx, var) \
        if (!mxIsDouble(prhs[idx]) && !mxIsSingle(prhs[idx])) \
            mexErrMsgIdAndTxt("load_slab_lz4_save_as_tif:type", #var " must be float scalar."); \
        float var = float(mxGetScalar(prhs[idx]));
    GRAB_FLOAT(5, scal)
    GRAB_FLOAT(6, ampl)
    GRAB_FLOAT(7, dmin)
    GRAB_FLOAT(8, dmax)

    // Compression string (default = 'deflate')
    std::string compression = "deflate";
    if (nrhs >= 10) {
        if (mxIsChar(prhs[9])) {
            char* s = mxArrayToUTF8String(prhs[9]);
            compression = s; mxFree(s);
        } else {
            mexErrMsgIdAndTxt("load_slab_lz4_save_as_tif:type", "compression must be a string.");
        }
    }
    if (compression != "none" && compression != "lzw" && compression != "deflate")
        mexErrMsgIdAndTxt("load_slab_lz4_save_as_tif:compression",
                          "compression must be 'none', 'lzw', or 'deflate'.");

    // useTiles logical (default = false)
    bool useTiles = false;
    if (nrhs >= 11) {
        if (mxIsLogical(prhs[10]) || mxIsNumeric(prhs[10]))
            useTiles = (mxGetScalar(prhs[10]) != 0);
        else
            mexErrMsgIdAndTxt("load_slab_lz4_save_as_tif:tiles", "useTiles must be logical or numeric.");
    }

    // maxThreads (default = hardware_concurrency)
    int maxThreads = std::thread::hardware_concurrency();
    if (nrhs >= 12) {
        if (!mxIsDouble(prhs[11]) && !mxIsUint32(prhs[11]))
            mexErrMsgIdAndTxt("load_slab_lz4_save_as_tif:type", "maxThreads must be integer.");
        maxThreads = int(mxGetScalar(prhs[11]));
        if (maxThreads < 1) maxThreads = 1;
    }

    // isXYZLayout (default = true)
    bool isXYZLayout = true;
    if (nrhs >= 13) {
        if (mxIsLogical(prhs[12]) || mxIsNumeric(prhs[12]))
            isXYZLayout = (mxGetScalar(prhs[12]) != 0);
        else
            mexErrMsgIdAndTxt("load_slab_lz4_save_as_tif:isXYZ", "isXYZLayout must be logical or numeric.");
    }

    // Output type (auto)
    mxClassID outType = (scal <= 255) ? mxUINT8_CLASS : mxUINT16_CLASS;
    if (outType != mxUINT8_CLASS && outType != mxUINT16_CLASS)
        mexErrMsgIdAndTxt("load_slab_lz4_save_as_tif:badType", "Output type must be uint8 or uint16.");

    // Get src and dst file lists (as std::string)
    std::vector<std::string> srcFiles(nBricks), dstFiles(nSlices);
    for (size_t i = 0; i < nBricks; ++i) {
        char* f = mxArrayToUTF8String(mxGetCell(prhs[0], i));
        srcFiles[i] = f; mxFree(f);
    }
    for (size_t i = 0; i < nSlices; ++i) {
        char* f = mxArrayToUTF8String(mxGetCell(prhs[1], i));
        dstFiles[i] = f; mxFree(f);
    }

    // Check output file parent directory exists & writable
    for (size_t i = 0; i < nSlices; ++i) {
        fs::path dir = fs::path(dstFiles[i]).parent_path();
        if (!dir.empty() && !fs::exists(dir))
            mexErrMsgIdAndTxt("load_slab_lz4_save_as_tif:dir", "Output directory does not exist: %s", dir.string().c_str());
        if (fs::exists(dstFiles[i]) && access(dstFiles[i].c_str(), W_OK) != 0)
            mexErrMsgIdAndTxt("load_slab_lz4_save_as_tif:writable", "Cannot overwrite read-only file: %s", dstFiles[i].c_str());
    }

    // Pass through p1/p2 for region spec (these are still MATLAB arrays)
    ValidatedInputs inp {
        std::move(srcFiles), std::move(dstFiles), std::move(dims),
        scal, ampl, dmin, dmax,
        compression, useTiles, maxThreads, outType,
        prhs[2], prhs[3], nBricks, nSlices, isXYZLayout
    };
    return inp;
}


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    try {
        // 1. Validate and parse all arguments (populate ValidatedInputs)
        ValidatedInputs inp = validate_inputs(prhs, nrhs);

        // Helper lambda to run slab load and save, and print timings
        auto load_and_save = [&](auto dummyType) {
            using T = decltype(dummyType);

            auto t0 = std::chrono::high_resolution_clock::now();
            auto slices = loadSlabLz4<T>(inp); // vector<unique_ptr<T[]>>
            auto t1 = std::chrono::high_resolution_clock::now();

            double dt_assemble = std::chrono::duration<double>(t1 - t0).count();
            mexPrintf("   slab assembled + scaled in %.1fs.\n", dt_assemble);

            std::vector<T*> rawPtrs;
            rawPtrs.reserve(slices.size());
            for (const auto& uptr : slices)
                rawPtrs.push_back(uptr.get());

            auto t2 = std::chrono::high_resolution_clock::now();
            saveSlabTif<T>(rawPtrs, inp);
            auto t3 = std::chrono::high_resolution_clock::now();

            double dt_save = std::chrono::duration<double>(t3 - t2).count();
            mexPrintf("   Saved %zu slices in %.1fs.\n", rawPtrs.size(), dt_save);
        };

        if (inp.outType == mxUINT8_CLASS) {
            load_and_save(uint8_t{});
        } else {
            load_and_save(uint16_t{});
        }

    } catch (const std::exception& ex) {
        mexErrMsgIdAndTxt("load_slab_lz4_save_as_tif:exception", ex.what());
    }
}
