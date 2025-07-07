/*==============================================================================
  save_bl_tif.cpp

  High-throughput multi-threaded TIFF Z-slice saver for 3D MATLAB volumes.
  Now with explicit async multi-file buffering and decoupled I/O via a producer-
  consumer queue.

  USAGE:
    save_bl_tif(volume, fileList, isXYZ, compression[, nThreads, useTiles, nIOThreads]);

  INPUTS:
    • volume      : 3D MATLAB array (uint8 or uint16), or 2D for single slice.
    • fileList    : 1×Z cell array of output filenames, one per Z-slice.
    • isXYZ       : Scalar logical/numeric. True if 'volume' is [X Y Z], false if [Y X Z].
    • compression : String. "none", "lzw", or "deflate".
    • nThreads    : (Optional) Number of worker threads. Default = half hardware concurrency.
    • useTiles    : (Optional) true for tiled TIFF output, false for classic strip mode.
    • nIOThreads  : (Optional, new) Number of async I/O threads. Default = 2.

  FEATURES:
    • Producer-consumer queue: compute and I/O phases run in parallel for high disk throughput.
    • Multi-threaded, atomic slice dispatch for maximum throughput.
    • Per-thread affinity for improved NUMA balancing.
    • Safe temp-file → rename for each slice (no partial writes).
    • Exception aggregation, robust error reporting.
    • Comments and structure for maintainability.

  DEPENDENCIES:
    • libtiff ≥ 4.7, MATLAB MEX API, C++17 <filesystem>, POSIX/Windows threading.

  AUTHOR:
    Keivan Moradi (with ChatGPT-4o assistance)

  LICENSE:
    GNU GPL v3 — https://www.gnu.org/licenses/gpl-3.0.html
==============================================================================*/

/*==============================================================================
  save_bl_tif.cpp

  High-throughput multi-threaded TIFF Z-slice saver for 3D MATLAB volumes.
  Now with async multi-file buffering, tile mode, advanced I/O flags,
  and finer NUMA/core affinity. All features are optional.

  USAGE:
    save_bl_tif(volume, fileList, isXYZ, compression,
                [, nThreads, useTiles, nIOThreads, useAdvancedIOFlags]);

  INPUTS:
    ...
    • useAdvancedIOFlags : (Optional, default false)
        Enable O_DIRECT (Linux) or FILE_FLAG_NO_BUFFERING (Windows) for TIFF file writing.
        Use only if you know your storage and alignment constraints.

  FEATURES:
    • Producer-consumer async I/O, robust tile/strip mode
    • Explicit O_DIRECT/FILE_FLAG_NO_BUFFERING support (optional)
    • NUMA/core-aware affinity (optional, per-thread)
    • Maintains code clarity, modularity, and error safety

  ...
==============================================================================*/

#include "mex.h"
#include "tiffio.h"
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
#include <system_error>
#if defined(_WIN32)
  #ifndef NOMINMAX
  #  define NOMINMAX
  #endif
  #include <windows.h>
  #include <bitset>
  #include <io.h>
  #ifndef W_OK
    #define W_OK 2
  #endif
  #define access _access
  #include <fcntl.h>
#elif defined(__linux__)
  #include <sched.h>
  #include <pthread.h>
  #include <unistd.h>
  #include <fcntl.h>
  #ifdef __has_include
    #if __has_include(<numa.h>)
      #include <numa.h>
    #endif
  #endif
#else
  #include <unistd.h>
#endif

namespace fs = std::filesystem;

// Advanced I/O alignment requirement (for O_DIRECT)
// Note: This is typically 4096 bytes, but can vary by device.
constexpr size_t IO_ALIGNMENT = 4096;

inline size_t get_available_cores() {
#if defined(__linux__)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n > 0) return static_cast<size_t>(n);
#elif defined(_WIN32)
    DWORD_PTR processMask = 0, systemMask = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask)) {
        return static_cast<size_t>(std::bitset<sizeof(processMask)*8>(processMask).count());
    }
#endif
    auto hint = std::thread::hardware_concurrency();
    return hint > 0 ? static_cast<size_t>(hint) : 1;
}

// ========================== NUMA/Core Affinity ===============================
inline void set_thread_affinity(size_t thread_index, size_t total_threads) {
#if defined(_WIN32)
    DWORD_PTR processMask = 0, systemMask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask)) {
        throw std::system_error(
            static_cast<int>(GetLastError()), std::system_category(), "GetProcessAffinityMask failed");
    }
    std::vector<DWORD> cpus;
    for (DWORD i = 0; i < sizeof(processMask) * 8; ++i) {
        if (processMask & (DWORD_PTR(1) << i))
            cpus.push_back(i);
    }
    if (cpus.empty()) cpus.push_back(0);
    DWORD core = cpus[ thread_index % cpus.size() ];
    DWORD_PTR mask = (DWORD_PTR(1) << core);
    if (SetThreadAffinityMask(GetCurrentThread(), mask) == 0) {
        throw std::system_error(
            static_cast<int>(GetLastError()), std::system_category(), "SetThreadAffinityMask failed");
    }
#elif defined(__linux__)
    long num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_cpus < 1) num_cpus = 1;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    // NUMA node interleaving: Even thread_index -> node 0, odd -> node 1 (if >1 node)
    #ifdef NUMA_VERSION1_COMPATIBILITY
    if (numa_available() != -1 && numa_max_node() > 0) {
        int node = thread_index % (numa_max_node() + 1);
        std::vector<int> cpus_on_node;
        struct bitmask *bm = numa_allocate_cpumask();
        numa_node_to_cpus(node, bm);
        for (int c = 0; c < num_cpus; ++c)
            if (numa_bitmask_isbitset(bm, c)) cpus_on_node.push_back(c);
        numa_free_cpumask(bm);
        int cpu = cpus_on_node[thread_index % cpus_on_node.size()];
        CPU_SET(cpu, &cpuset);
    } else
    #endif
    {
        CPU_SET(thread_index % num_cpus, &cpuset);
    }
    int err = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (err != 0) {
        throw std::system_error(err, std::generic_category(), "pthread_setaffinity_np failed");
    }
#else
    (void)thread_index; (void)total_threads;
#endif
}

// ================ RAII TIFF Writer (with O_DIRECT support) ===================
struct TiffWriter {
    TIFF* tif;
#if defined(_WIN32)
    HANDLE rawHandle = INVALID_HANDLE_VALUE;
#elif defined(__linux__)
    int rawFd = -1;
#endif
    TiffWriter(const std::string& path, const char* mode, bool useAdvancedIOFlags) {
        tif = nullptr;
#if defined(_WIN32)
        if (useAdvancedIOFlags) {
            rawHandle = CreateFileA(
                path.c_str(),
                GENERIC_WRITE,
                0,
                NULL,
                CREATE_ALWAYS,
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING,
                NULL);
            if (rawHandle == INVALID_HANDLE_VALUE)
                throw std::runtime_error("Cannot open TIFF with FILE_FLAG_NO_BUFFERING: " + path);
            tif = TIFFFdOpen(_open_osfhandle((intptr_t)rawHandle, _O_RDWR), path.c_str(), mode);
        }
        else
#endif
#if defined(__linux__)
        if (useAdvancedIOFlags) {
            rawFd = ::open(path.c_str(), O_WRONLY|O_CREAT|O_TRUNC|O_DIRECT, 0666);
            if (rawFd < 0)
                throw std::runtime_error("Cannot open TIFF with O_DIRECT: " + path);
            tif = TIFFFdOpen(rawFd, path.c_str(), mode);
        }
        else
#endif
        {
            tif = TIFFOpen(path.c_str(), mode);
        }
        if (!tif)
            throw std::runtime_error("Cannot open TIFF for writing: " + path);
    }
    ~TiffWriter() { if (tif) TIFFClose(tif); }
};

// ========== Tile Size Selection Helper =======================================
inline void select_tile_size(uint32_t width, uint32_t height, uint32_t &tileWidth, uint32_t &tileLength) {
    if (width >= 1024 && height >= 1024) { tileWidth = 1024; tileLength = 1024; }
    else if (width >= 512 && height >= 512) { tileWidth = 512; tileLength = 512; }
    else { tileWidth = 256; tileLength = 256; }
}

// ======================= Producer-Consumer Slice Work ========================
struct SliceWork {
    std::vector<uint8_t> prepared_slice_data; // Buffer (row- or tile-major)
    uint32_t slice_index;
    std::string output_path;
    uint32_t width_dim, height_dim, bytes_per_pixel;
    uint16_t compression_type;
    bool use_tiles;
    bool is_xyz;
    uint32_t tile_width, tile_length;
};

class SliceWorkQueue {
    std::queue<SliceWork> queue_;
    std::mutex mtx_;
    std::condition_variable cv_not_empty_, cv_not_full_;
    const size_t max_queue_size_;
    bool finished_ = false;
public:
    SliceWorkQueue(size_t max_size) : max_queue_size_(max_size) {}

    void push(SliceWork&& work) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_not_full_.wait(lk, [&] { return queue_.size() < max_queue_size_ || finished_; });
        if (finished_) return;
        queue_.push(std::move(work));
        cv_not_empty_.notify_one();
    }
    bool pop(SliceWork& work) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_not_empty_.wait(lk, [&] { return !queue_.empty() || finished_; });
        if (queue_.empty()) return false;
        work = std::move(queue_.front());
        queue_.pop();
        cv_not_full_.notify_one();
        return true;
    }
    void mark_finished() {
        std::lock_guard<std::mutex> lk(mtx_);
        finished_ = true;
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }
};

// ===================== Tile & Strip Buffer Preparation =======================

static void prepare_and_enqueue_slice(
    const uint8_t*      volume_data,
    uint32_t            slice_index,
    uint32_t            width_dim,
    uint32_t            height_dim,
    uint32_t            num_slices,
    uint32_t            bytes_per_pixel,
    bool                is_xyz,
    const std::string&  output_path,
    uint16_t            compression_type,
    bool                use_tiles,
    SliceWorkQueue&     io_queue
) {
    uint32_t tileWidth=0, tileLength=0;
    if (use_tiles) select_tile_size(width_dim, height_dim, tileWidth, tileLength);
    const size_t slice_byte_count = size_t(width_dim) * size_t(height_dim) * size_t(bytes_per_pixel);

    std::vector<uint8_t> prepared_slice_data;
    if (!use_tiles) {
        // STRIP: always prepare row-major buffer for the slice
        prepared_slice_data.resize(slice_byte_count);
        if (is_xyz) {
            const uint8_t* src_ptr = volume_data + size_t(slice_index) * slice_byte_count;
            std::memcpy(prepared_slice_data.data(), src_ptr, slice_byte_count);
        } else {
            for (uint32_t row = 0; row < height_dim; ++row) {
                for (uint32_t col = 0; col < width_dim; ++col) {
                    size_t src_offset = (size_t(row) + size_t(col) * size_t(height_dim)) * size_t(bytes_per_pixel);
                    size_t dst_offset = (size_t(row) * size_t(width_dim) + size_t(col)) * size_t(bytes_per_pixel);
                    std::memcpy(&prepared_slice_data[dst_offset],
                                volume_data + size_t(slice_index) * size_t(height_dim) * size_t(width_dim) * size_t(bytes_per_pixel) + src_offset,
                                size_t(bytes_per_pixel));
                }
            }
        }
    } else {
        // TILE: prepare buffer tile-row-major
        const uint32_t tilesAcross  = (width_dim  + tileWidth  - 1) / tileWidth;
        const uint32_t tilesDown    = (height_dim + tileLength - 1) / tileLength;
        const size_t   tileBytes    = size_t(tileWidth) * size_t(tileLength) * size_t(bytes_per_pixel);
        prepared_slice_data.resize(tilesAcross * tilesDown * tileBytes, 0);
        for (uint32_t tileRowIndex = 0; tileRowIndex < tilesDown; ++tileRowIndex) {
            for (uint32_t tileColIndex = 0; tileColIndex < tilesAcross; ++tileColIndex) {
                uint32_t rowStart    = tileRowIndex * tileLength;
                uint32_t rowsToWrite = std::min(tileLength, height_dim - rowStart);
                uint32_t colStart    = tileColIndex * tileWidth;
                uint32_t colsToWrite = std::min(tileWidth,  width_dim  - colStart);

                uint8_t* tilePtr = prepared_slice_data.data() +
                    (tileRowIndex * tilesAcross + tileColIndex) * tileBytes;
                std::fill(tilePtr, tilePtr + tileBytes, 0);

                if (is_xyz) {
                    for (uint32_t rowInTile = 0; rowInTile < rowsToWrite; ++rowInTile) {
                        size_t srcOffset = (size_t(rowStart) + size_t(rowInTile)) * size_t(width_dim) * size_t(bytes_per_pixel)
                                         + size_t(colStart) * size_t(bytes_per_pixel)
                                         + size_t(slice_index) * slice_byte_count;
                        size_t dstOffset = size_t(rowInTile) * size_t(tileWidth) * size_t(bytes_per_pixel);
                        std::memcpy(tilePtr + dstOffset,
                                    volume_data + srcOffset, size_t(colsToWrite) * size_t(bytes_per_pixel));
                    }
                } else {
                    for (uint32_t rowInTile = 0; rowInTile < rowsToWrite; ++rowInTile) {
                        for (uint32_t colInTile = 0; colInTile < colsToWrite; ++colInTile) {
                            size_t srcOffset = (size_t(rowStart + rowInTile) + size_t(colStart + colInTile) * size_t(height_dim)) * size_t(bytes_per_pixel)
                                             + size_t(slice_index) * size_t(height_dim) * size_t(width_dim) * size_t(bytes_per_pixel);
                            size_t dstOffset = (size_t(rowInTile) * size_t(tileWidth) + size_t(colInTile)) * size_t(bytes_per_pixel);
                            std::memcpy(tilePtr + dstOffset,
                                        volume_data + srcOffset, size_t(bytes_per_pixel));
                        }
                    }
                }
            }
        }
    }
    io_queue.push(SliceWork{
        std::move(prepared_slice_data),
        slice_index, output_path, width_dim, height_dim, bytes_per_pixel,
        compression_type, use_tiles, is_xyz, tileWidth, tileLength
    });
}

// =========== Core TIFF Slice Write, Tile/Strip, O_DIRECT aware ===============

static void write_slice_to_tiff_buffer(
    const SliceWork& sliceWork,
    bool useAdvancedIOFlags
) {
    fs::path temp_file = fs::path(sliceWork.output_path).concat(".tmp");
    {
        TiffWriter writer(temp_file.string(), "w", useAdvancedIOFlags);
        TIFF* tif = writer.tif;
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,      sliceWork.width_dim);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH,     sliceWork.height_dim);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,   static_cast<uint16_t>(sliceWork.bytes_per_pixel * 8));
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, static_cast<uint16_t>(1));
        TIFFSetField(tif, TIFFTAG_COMPRESSION,     sliceWork.compression_type);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);

        if (sliceWork.compression_type == COMPRESSION_ADOBE_DEFLATE) {
            const int zipLevel = 1;
            TIFFSetField(tif, TIFFTAG_ZIPQUALITY, zipLevel);
            TIFFSetField(tif, TIFFTAG_PREDICTOR, 1);
        }

        if (sliceWork.use_tiles) {
            TIFFSetField(tif, TIFFTAG_TILEWIDTH, sliceWork.tile_width);
            TIFFSetField(tif, TIFFTAG_TILELENGTH, sliceWork.tile_length);
            const uint32_t tilesAcross  = (sliceWork.width_dim  + sliceWork.tile_width  - 1) / sliceWork.tile_width;
            const uint32_t tilesDown    = (sliceWork.height_dim + sliceWork.tile_length - 1) / sliceWork.tile_length;
            const size_t   tileBytes    = size_t(sliceWork.tile_width) * size_t(sliceWork.tile_length) * size_t(sliceWork.bytes_per_pixel);

            for (uint32_t tileRowIndex = 0; tileRowIndex < tilesDown; ++tileRowIndex) {
                for (uint32_t tileColIndex = 0; tileColIndex < tilesAcross; ++tileColIndex) {
                    uint8_t* tilePtr = const_cast<uint8_t*>(sliceWork.prepared_slice_data.data())
                        + (tileRowIndex * tilesAcross + tileColIndex) * tileBytes;
                    tstrip_t tileIdx = TIFFComputeTile(tif, tileColIndex * sliceWork.tile_width, tileRowIndex * sliceWork.tile_length, 0, 0);
                    if (TIFFWriteEncodedTile(tif, tileIdx, tilePtr, tileBytes) < 0)
                        throw std::runtime_error("TIFF tile write failed at (" +
                            std::to_string(tileColIndex) + "," + std::to_string(tileRowIndex) + ")");
                }
            }
        } else {
            TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, sliceWork.height_dim);
            const size_t byte_count = size_t(sliceWork.width_dim) * size_t(sliceWork.height_dim) * size_t(sliceWork.bytes_per_pixel);
            void* buf = const_cast<void*>(static_cast<const void*>(sliceWork.prepared_slice_data.data()));
            if (TIFFWriteEncodedStrip(tif, 0, buf, byte_count) < 0)
                throw std::runtime_error("TIFF write failed on strip for " + sliceWork.output_path);
        }
    }
    // Atomic rename: temp file → final output path
    std::error_code ec;
    fs::rename(temp_file, sliceWork.output_path, ec);
    if (ec) {
        if (fs::exists(sliceWork.output_path)) fs::remove(sliceWork.output_path);
        fs::rename(temp_file, sliceWork.output_path, ec);
        if (ec)
            throw std::runtime_error("Failed to rename " + temp_file.string() + " → " + sliceWork.output_path);
    }
}

// ======================= MATLAB MEX Entry Point (Main) =======================
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try {
        if (nrhs < 4 || nrhs > 8)
            mexErrMsgIdAndTxt("save_bl_tif:usage",
                "Usage: save_bl_tif(vol, fileList, isXYZ, compression[, nThreads, useTiles, nIOThreads, useAdvancedIOFlags]);");

        if (!mxIsUint8(prhs[0]) && !mxIsUint16(prhs[0]))
            mexErrMsgIdAndTxt("save_bl_tif:type", "Volume must be uint8 or uint16.");

        const mwSize*  raw_dims   = mxGetDimensions(prhs[0]);
        const uint32_t raw_rows   = static_cast<uint32_t>(raw_dims[0]);
        const uint32_t raw_cols   = static_cast<uint32_t>(raw_dims[1]);
        const uint32_t num_slices = (mxGetNumberOfDimensions(prhs[0]) == 3 ? static_cast<uint32_t>(raw_dims[2]) : 1);

        const bool is_xyz = mxIsLogicalScalarTrue(prhs[2]) || (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0);
        const uint32_t width_dim  = is_xyz ? raw_rows : raw_cols;
        const uint32_t height_dim = is_xyz ? raw_cols : raw_rows;

        char* comp_cstr = mxArrayToUTF8String(prhs[3]);
        std::string compression_str(comp_cstr);
        mxFree(comp_cstr);
        uint16_t compression_type =
               compression_str == "none"    ? COMPRESSION_NONE
             : compression_str == "lzw"     ? COMPRESSION_LZW
             : compression_str == "deflate" ? COMPRESSION_ADOBE_DEFLATE
             : throw std::runtime_error("Invalid compression: " + compression_str);

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != num_slices)
            mexErrMsgIdAndTxt("save_bl_tif:files",
                "fileList must be a cell-array of length = number of slices.");
        std::vector<std::string> output_paths(num_slices);
        for (uint32_t i = 0; i < num_slices; ++i) {
            char* s = mxArrayToUTF8String(mxGetCell(prhs[1], i));
            output_paths[i] = s;
            mxFree(s);
        }

        for (auto& path : output_paths) {
            fs::path dir = fs::path(path).parent_path();
            if (!dir.empty() && !fs::exists(dir))
                mexErrMsgIdAndTxt("save_bl_tif:invalidPath",
                    "Directory does not exist: %s", dir.string().c_str());
            if (fs::exists(path) && access(path.c_str(), W_OK) != 0)
                mexErrMsgIdAndTxt("save_bl_tif:readonly",
                    "Cannot overwrite read-only file: %s", path.c_str());
        }

        const uint8_t* volume_data    = static_cast<const uint8_t*>(mxGetData(prhs[0]));
        const uint32_t bytes_per_pixel = (mxGetClassID(prhs[0]) == mxUINT16_CLASS ? 2u : 1u);

        const size_t hw_cores         = get_available_cores();
        const size_t safe_cores       = hw_cores ? hw_cores : 1;
        const size_t default_threads  = std::max(safe_cores / 2, size_t(1));
        const size_t requested_threads = (nrhs >= 5 && !mxIsEmpty(prhs[4])? static_cast<size_t>(mxGetScalar(prhs[4])) : default_threads);
        const size_t computation_thread_count = std::min(requested_threads, static_cast<size_t>(num_slices));
        const bool use_tiles = (nrhs >= 6) ? (mxIsLogicalScalarTrue(prhs[5]) || (mxIsNumeric(prhs[5]) && mxGetScalar(prhs[5]) != 0)) : false;
        const size_t io_thread_count = (nrhs >= 7 && !mxIsEmpty(prhs[6])) ? static_cast<size_t>(mxGetScalar(prhs[6])) : 2;
        const bool useAdvancedIOFlags = (nrhs >= 8 && !mxIsEmpty(prhs[7])) ?
            (mxIsLogicalScalarTrue(prhs[7]) || (mxIsNumeric(prhs[7]) && mxGetScalar(prhs[7]) != 0)) : false;
        const size_t bounded_queue_size = computation_thread_count + io_thread_count + 4;

        std::vector<std::string> errors;
        std::mutex error_mutex;

        SliceWorkQueue io_queue(bounded_queue_size);
        std::atomic<uint32_t> next_slice{0};

        std::vector<std::thread> io_workers;
        io_workers.reserve(io_thread_count);
        for (size_t io_thread_index = 0; io_thread_index < io_thread_count; ++io_thread_index) {
            io_workers.emplace_back([&, io_thread_index]() {
                try { set_thread_affinity(io_thread_index, io_thread_count); }
                catch (const std::system_error& ex) {
                    std::lock_guard<std::mutex> lg(error_mutex);
                    errors.push_back(ex.what());
                    return;
                }
                SliceWork slice_work;
                while (io_queue.pop(slice_work)) {
                    try {
                        write_slice_to_tiff_buffer(slice_work, useAdvancedIOFlags);
                    } catch (const std::exception& ex) {
                        std::lock_guard<std::mutex> lg(error_mutex);
                        errors.push_back("I/O thread failed for " + slice_work.output_path + ": " + ex.what());
                    }
                }
            });
        }

        std::vector<std::thread> computation_workers;
        computation_workers.reserve(computation_thread_count);
        for (size_t computation_thread_index = 0; computation_thread_index < computation_thread_count; ++computation_thread_index) {
            computation_workers.emplace_back([&, computation_thread_index]() {
                try { set_thread_affinity(computation_thread_index, computation_thread_count); }
                catch (const std::system_error& ex) {
                    std::lock_guard<std::mutex> lg(error_mutex);
                    errors.push_back(ex.what());
                    return;
                }
                const size_t slices_per_dispatch = 4;
                while (true) {
                    uint32_t start = next_slice.fetch_add(
                        static_cast<uint32_t>(slices_per_dispatch),
                        std::memory_order_relaxed
                    );
                    if (start >= num_slices) break;
                    uint32_t end = std::min(num_slices, start + static_cast<uint32_t>(slices_per_dispatch));
                    for (uint32_t idx = start; idx < end; ++idx) {
                        try {
                            prepare_and_enqueue_slice(
                                volume_data, idx,
                                width_dim, height_dim, num_slices, bytes_per_pixel,
                                is_xyz, output_paths[idx], compression_type, use_tiles,
                                io_queue
                            );
                        } catch (const std::exception& ex) {
                            std::lock_guard<std::mutex> lg(error_mutex);
                            errors.push_back(
                                "Producer thread failed for " + output_paths[idx] + ": " + ex.what());
                            return;
                        }
                    }
                }
            });
        }

        for (auto& th : computation_workers) th.join();
        io_queue.mark_finished();
        for (auto& th : io_workers) th.join();

        if (!errors.empty())
            mexErrMsgIdAndTxt("save_bl_tif:runtime", errors.front().c_str());
        if (nlhs > 0)
            plhs[0] = const_cast<mxArray*>(prhs[0]);
    }
    catch (const std::exception& ex) {
        mexErrMsgIdAndTxt("save_bl_tif:runtime", ex.what());
    }
    catch (...) {
        mexErrMsgIdAndTxt("save_bl_tif:unknown", "Unknown error in save_bl_tif");
    }
}
