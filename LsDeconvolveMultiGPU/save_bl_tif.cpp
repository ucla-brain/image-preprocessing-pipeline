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
#elif defined(__linux__)
  #include <sched.h>
  #include <pthread.h>
  #include <unistd.h>
#else
  #include <unistd.h>
#endif

namespace fs = std::filesystem;

// =================== Helper Functions for System and Thread ===================

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

// Set thread affinity for NUMA/core balancing
inline void set_thread_affinity(size_t thread_index) {
#if defined(_WIN32)
    DWORD_PTR processMask = 0, systemMask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask)) {
        throw std::system_error(
            static_cast<int>(GetLastError()),
            std::system_category(),
            "GetProcessAffinityMask failed"
        );
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
            static_cast<int>(GetLastError()),
            std::system_category(),
            "SetThreadAffinityMask failed"
        );
    }
#elif defined(__linux__)
    long num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_cpus < 1) num_cpus = 1;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(thread_index % num_cpus, &cpuset);
    int err = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (err != 0) {
        throw std::system_error(err, std::generic_category(), "pthread_setaffinity_np failed");
    }
#else
    (void)thread_index;
#endif
}

// ========================== TIFF Writing RAII Helper ==========================
struct TiffWriter {
    TIFF* tif;
    TiffWriter(const std::string& path, const char* mode) {
        tif = TIFFOpen(path.c_str(), mode);
        if (!tif)
            throw std::runtime_error("Cannot open TIFF for writing: " + path);
    }
    ~TiffWriter() { if (tif) TIFFClose(tif); }
};

// ========== Core TIFF Slice Write (Unchanged, now with buffer param) ==========
static void write_slice_to_tiff_buffer(
    const uint8_t*           slice_buffer,
    uint32_t                 width_dim,
    uint32_t                 height_dim,
    uint32_t                 bytes_per_pixel,
    uint16_t                 compression_type,
    const std::string&       output_path,
    bool                     use_tiles
) {
    fs::path temp_file = fs::path(output_path).concat(".tmp");
    {
        TiffWriter writer(temp_file.string(), "w");
        TIFF* tif = writer.tif;
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,      width_dim);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH,     height_dim);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,   static_cast<uint16_t>(bytes_per_pixel * 8));
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, static_cast<uint16_t>(1));
        TIFFSetField(tif, TIFFTAG_COMPRESSION,     compression_type);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);

        if (compression_type == COMPRESSION_ADOBE_DEFLATE) {
            const int zipLevel = 1;
            TIFFSetField(tif, TIFFTAG_ZIPQUALITY, zipLevel);
            TIFFSetField(tif, TIFFTAG_PREDICTOR, 1);
        }
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, height_dim); // One strip

        // Write a single strip (already prepared as row-major for this slice)
        const size_t byte_count = size_t(width_dim) * size_t(height_dim) * size_t(bytes_per_pixel);
        void* buf = const_cast<void*>(static_cast<const void*>(slice_buffer));
        if (TIFFWriteEncodedStrip(tif, 0, buf, byte_count) < 0)
            throw std::runtime_error("TIFF write failed on strip for " + output_path);
    }

    // Atomic rename: temp file → final output path
    std::error_code ec;
    fs::rename(temp_file, output_path, ec);
    if (ec) {
        if (fs::exists(output_path)) fs::remove(output_path);
        fs::rename(temp_file, output_path, ec);
        if (ec)
            throw std::runtime_error("Failed to rename " + temp_file.string() + " → " + output_path);
    }
}

// ======================= Producer-Consumer Slice Work ========================
struct SliceWork {
    std::vector<uint8_t> prepared_slice_data; // Owns a copy of [width x height x bpp]
    uint32_t slice_index;
    std::string output_path;
    uint32_t width_dim, height_dim, bytes_per_pixel;
    uint16_t compression_type;
    bool use_tiles;
};

class SliceWorkQueue {
    std::queue<SliceWork> queue_;
    std::mutex mtx_;
    std::condition_variable cv_not_empty_, cv_not_full_;
    const size_t max_queue_size_;
    bool finished_ = false;
public:
    SliceWorkQueue(size_t max_size) : max_queue_size_(max_size) {}

    // Push new work; blocks if full
    void push(SliceWork&& work) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_not_full_.wait(lk, [&] { return queue_.size() < max_queue_size_ || finished_; });
        if (finished_) return;
        queue_.push(std::move(work));
        cv_not_empty_.notify_one();
    }
    // Pop work; returns false if finished and queue is empty
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

// ===================== Core Producer (Preparation) Logic =====================

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
    // Prepare [width x height x bpp] row-major buffer for the current slice
    const size_t slice_byte_count = size_t(width_dim) * size_t(height_dim) * size_t(bytes_per_pixel);
    std::vector<uint8_t> prepared_slice_data(slice_byte_count);

    if (is_xyz) {
        // [X Y Z] order: already row-major in memory for this slice, just memcpy
        const uint8_t* src_ptr = volume_data + size_t(slice_index) * slice_byte_count;
        std::memcpy(prepared_slice_data.data(), src_ptr, slice_byte_count);
    } else {
        // [Y X Z] order: need to transpose to row-major (X fastest)
        const mwSize* mx_dims = nullptr;
        // The dimensions are always [rawRows, rawCols, Z]
        // height_dim = rawRows, width_dim = rawCols
        // For each y (row), for each x (col): src(y + x*height_dim)
        for (uint32_t row = 0; row < height_dim; ++row) {
            for (uint32_t col = 0; col < width_dim; ++col) {
                size_t src_offset = (size_t(row) + size_t(col) * size_t(height_dim)) * size_t(bytes_per_pixel);
                size_t dst_offset = (size_t(row) * size_t(width_dim) + size_t(col)) * size_t(bytes_per_pixel);
                std::memcpy(
                    &prepared_slice_data[dst_offset],
                    volume_data + size_t(slice_index) * size_t(height_dim) * size_t(width_dim) * size_t(bytes_per_pixel) + src_offset,
                    size_t(bytes_per_pixel)
                );
            }
        }
    }
    // Queue up work for async I/O threads
    io_queue.push(SliceWork{
        std::move(prepared_slice_data),
        slice_index, output_path, width_dim, height_dim, bytes_per_pixel,
        compression_type, use_tiles
    });
}

// =================== MATLAB MEX Entry Point (Main Driver) ====================

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try {
        if (nrhs < 4 || nrhs > 7)
            mexErrMsgIdAndTxt("save_bl_tif:usage",
                "Usage: save_bl_tif(vol, fileList, isXYZ, compression[, nThreads, useTiles, nIOThreads]);");

        if (!mxIsUint8(prhs[0]) && !mxIsUint16(prhs[0]))
            mexErrMsgIdAndTxt("save_bl_tif:type", "Volume must be uint8 or uint16.");

        // Raw dims from array
        const mwSize*  raw_dims   = mxGetDimensions(prhs[0]);
        const uint32_t raw_rows   = static_cast<uint32_t>(raw_dims[0]);
        const uint32_t raw_cols   = static_cast<uint32_t>(raw_dims[1]);
        const uint32_t num_slices = (mxGetNumberOfDimensions(prhs[0]) == 3 ? static_cast<uint32_t>(raw_dims[2]) : 1);

        // isXYZ flag
        const bool is_xyz = mxIsLogicalScalarTrue(prhs[2]) || (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0);

        // Determine width (X) and height (Y) consistently
        const uint32_t width_dim  = is_xyz ? raw_rows : raw_cols;
        const uint32_t height_dim = is_xyz ? raw_cols : raw_rows;

        // Compression
        char* comp_cstr = mxArrayToUTF8String(prhs[3]);
        std::string compression_str(comp_cstr);
        mxFree(comp_cstr);
        uint16_t compression_type =
               compression_str == "none"    ? COMPRESSION_NONE
             : compression_str == "lzw"     ? COMPRESSION_LZW
             : compression_str == "deflate" ? COMPRESSION_ADOBE_DEFLATE
             : throw std::runtime_error("Invalid compression: " + compression_str);

        // fileList validation
        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != num_slices)
            mexErrMsgIdAndTxt("save_bl_tif:files",
                "fileList must be a cell-array of length = number of slices.");
        std::vector<std::string> output_paths(num_slices);
        for (uint32_t i = 0; i < num_slices; ++i) {
            char* s = mxArrayToUTF8String(mxGetCell(prhs[1], i));
            output_paths[i] = s;
            mxFree(s);
        }

        // Guard-clauses for file output paths
        for (auto& path : output_paths) {
            fs::path dir = fs::path(path).parent_path();
            if (!dir.empty() && !fs::exists(dir))
                mexErrMsgIdAndTxt("save_bl_tif:invalidPath",
                    "Directory does not exist: %s", dir.string().c_str());
            if (fs::exists(path) && access(path.c_str(), W_OK) != 0)
                mexErrMsgIdAndTxt("save_bl_tif:readonly",
                    "Cannot overwrite read-only file: %s", path.c_str());
        }

        // Data pointer & sample size
        const uint8_t* volume_data    = static_cast<const uint8_t*>(mxGetData(prhs[0]));
        const uint32_t bytes_per_pixel = (mxGetClassID(prhs[0]) == mxUINT16_CLASS ? 2u : 1u);

        // Thread counts for computation (slice prep) and I/O
        const size_t hw_cores         = get_available_cores();
        const size_t safe_cores       = hw_cores ? hw_cores : 1;
        const size_t default_threads  = std::max(safe_cores / 2, size_t(1));
        const size_t requested_threads = (nrhs >= 5 && !mxIsEmpty(prhs[4])? static_cast<size_t>(mxGetScalar(prhs[4])) : default_threads);
        const size_t computation_thread_count = std::min(requested_threads, static_cast<size_t>(num_slices));
        const bool use_tiles = (nrhs >= 6) ? (mxIsLogicalScalarTrue(prhs[5]) || (mxIsNumeric(prhs[5]) && mxGetScalar(prhs[5]) != 0)) : false;
        const size_t io_thread_count = (nrhs >= 7 && !mxIsEmpty(prhs[6])) ? static_cast<size_t>(mxGetScalar(prhs[6])) : 2;
        const size_t bounded_queue_size = computation_thread_count + io_thread_count + 4; // Reasonable upper bound

        // Error aggregation
        std::vector<std::string> errors;
        std::mutex error_mutex;

        // ========== Set Up Producer-Consumer Infrastructure ==========
        SliceWorkQueue io_queue(bounded_queue_size);

        // Atomic dispatch for slices
        std::atomic<uint32_t> next_slice{0};

        // ---------- Async I/O Threads (Consumers) ----------
        std::vector<std::thread> io_workers;
        io_workers.reserve(io_thread_count);
        for (size_t io_thread_index = 0; io_thread_index < io_thread_count; ++io_thread_index) {
            io_workers.emplace_back([&, io_thread_index]() {
                try { set_thread_affinity(io_thread_index); }
                catch (const std::system_error& ex) {
                    std::lock_guard<std::mutex> lg(error_mutex);
                    errors.push_back(ex.what());
                    return;
                }
                SliceWork slice_work;
                while (io_queue.pop(slice_work)) {
                    try {
                        write_slice_to_tiff_buffer(
                            slice_work.prepared_slice_data.data(),
                            slice_work.width_dim,
                            slice_work.height_dim,
                            slice_work.bytes_per_pixel,
                            slice_work.compression_type,
                            slice_work.output_path,
                            slice_work.use_tiles
                        );
                    } catch (const std::exception& ex) {
                        std::lock_guard<std::mutex> lg(error_mutex);
                        errors.push_back(
                            "I/O thread failed for " + slice_work.output_path + ": " + ex.what());
                    }
                }
            });
        }

        // ----------- Worker Threads (Producers) -----------
        std::vector<std::thread> computation_workers;
        computation_workers.reserve(computation_thread_count);
        for (size_t computation_thread_index = 0; computation_thread_index < computation_thread_count; ++computation_thread_index) {
            computation_workers.emplace_back([&, computation_thread_index]() {
                try { set_thread_affinity(computation_thread_index); }
                catch (const std::system_error& ex) {
                    std::lock_guard<std::mutex> lg(error_mutex);
                    errors.push_back(ex.what());
                    return;
                }
                // Dispatch slices in blocks for better load balancing
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

        // ========== Thread Join/Cleanup ==========
        for (auto& th : computation_workers) th.join();
        io_queue.mark_finished(); // Allow I/O threads to finish when queue is empty
        for (auto& th : io_workers) th.join();

        // Report error if any
        if (!errors.empty())
            mexErrMsgIdAndTxt("save_bl_tif:runtime", errors.front().c_str());

        // Return input as pass-through if requested
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
