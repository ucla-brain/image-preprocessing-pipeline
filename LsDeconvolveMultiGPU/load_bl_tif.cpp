/*==============================================================================
  load_bl_tif.cpp
  ---------------------------------------------------------------------------
  High-throughput sub-region loader for 3-D TIFF stacks (one TIFF per Z-slice)

  Author:       Keivan Moradi
  Code review:  ChatGPT (4-o, o3, 4.1)
  License:      GNU General Public License v3.0 (https://www.gnu.org/licenses/)

  ──────────────────────────────────────────────────────────────────────────────
  OVERVIEW
  --------
  • Purpose
      Efficiently extracts an X-Y subregion (ROI) from a series of grayscale
      TIFF slices (1 TIFF per Z) and assembles them into a 3D MATLAB array.
      Optimized for speed, reliability, and robustness in high-throughput,
      multi-core environments.

  • Highlights
      – Supports 8-bit and 16-bit grayscale TIFFs (single-channel).
      – Handles both tiled and stripped formats, including compression (e.g., Deflate, LZW).
      – Fully cross-platform: Windows, Linux, macOS (requires libtiff ≥ 4.0).
      – Uses modern C++14/17 and multi-threading with a dynamic work queue.
      – ROI coordinates and dimensions use `uint32_t` for clarity and safety.
      – Each thread independently opens its own TIFF and manages its own decode buffers.
      – Endianness correction (byte-swapping) is automatically handled by libtiff.
      – Output shape is [Y X Z] by default; optional transpose to [X Y Z].
      – ROI bounds are strictly validated across all slices before memory allocation.
      – All thread-level errors are aggregated and reported as a single message.

  ──────────────────────────────────────────────────────────────────────────────
  MATLAB USAGE
  ------------
      img = load_bl_tif(files, y, x, height, width [, transposeFlag]);

      • files          – 1×N cell array of full path strings (one per Z slice)
      • y, x           – 1-based upper-left ROI coordinate (double scalars)
      • height, width  – ROI dimensions in pixels (double scalars)
      • transposeFlag  – (optional) logical or uint32 scalar, default = false
                         If true, output is returned in [X Y Z] format

      • returns
        – img          – MATLAB array of class uint8 or uint16:
                          [height width Z] or [width height Z] if transposed

      Example:
          files = dir('/some/folder/*.tif');
          paths = fullfile({files.folder}, {files.name});
          blk   = load_bl_tif(paths, 101, 201, 512, 512);         % Standard
          blkT  = load_bl_tif(paths, 101, 201, 512, 512, true);   % Transposed

  ──────────────────────────────────────────────────────────────────────────────
  COMPILATION
  -----------
      • Recommended: MATLAB R2018a+ with a C++14/17-capable compiler.

      • Use the provided `build_mex.m` script, or compile manually:

          mex -R2018a -largeArrayDims CXXFLAGS="\$(CXXFLAGS) -std=c++17" \
              LDFLAGS="\$(LDFLAGS) -ltiff" load_bl_tif.cpp

      • Ensure libtiff headers and libraries are available to the compiler.
      • On Windows, link against a precompiled `tiff.lib`.

  ──────────────────────────────────────────────────────────────────────────────
  CONSTRAINTS & SAFEGUARDS
  -------------------------
      • Files must be sorted by Z; no sorting is performed internally.
      • All slices must share identical size, bit depth, and be grayscale (1 sample/pixel).
      • The ROI must lie fully inside each slice. Validation is strict and occurs before allocation.
      • The output array must not exceed 2,147,483,647 pixels per slice (MATLAB limit).
      • RGB or multi-channel TIFFs are not supported.

  ──────────────────────────────────────────────────────────────────────────────
  PARALLELISM & PERFORMANCE
  --------------------------
      • Uses a dynamic work queue to assign TIFF slices to threads.
      • Each thread uses independent TIFF handles and decode buffers.
      • Byte-swapping is delegated to libtiff, which transparently returns native-endian data.
      • Aggregates all error messages after the parallel phase finishes.
      • For best performance, store TIFFs on SSD/NVMe and avoid over-threading.

  ---------------------------------------------------------------------------
  © 2025 Keivan Moradi — Released under GPLv3. See LICENSE or visit:
                         https://www.gnu.org/licenses/gpl-3.0.html
==============================================================================*/

/*==============================================================================
  load_bl_tif.cpp
  ---------------------------------------------------------------------------
  High-throughput sub-region loader for 3-D TIFF stacks (one TIFF per Z-slice)
  Author:       Keivan Moradi
  Code review:  ChatGPT (4-o, o3, 4.1)
  License:      GNU General Public License v3.0 (https://www.gnu.org/licenses/)
==============================================================================*/

/*==============================================================================
  load_bl_tif.cpp
  ---------------------------------------------------------------------------
  High-throughput sub-region loader for 3-D TIFF stacks (one TIFF per Z-slice)
  Author:       Keivan Moradi
  Code review:  ChatGPT (4-o, o3, 4.1)
  License:      GNU GPL v3.0 (https://www.gnu.org/licenses/)
==============================================================================*/

#define NOMINMAX
#include "mex.h"
#include "matrix.h"
#include "tiffio.h"
#include "mex_thread_utils.hpp"

#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <exception>
#include <atomic>
#include <sstream>
#include <limits>
#include <cstring>

// Supported bit depths and MATLAB limits
static constexpr uint16_t kSupportedBitDepth8  = 8;
static constexpr uint16_t kSupportedBitDepth16 = 16;
static constexpr size_t   kMaxPixelsPerSlice   = static_cast<size_t>(std::numeric_limits<int>::max());
static constexpr size_t   kWires               = 1;

// RAII wrapper for mxArrayToUTF8String()
struct MatlabString {
    char* ptr;
    explicit MatlabString(const mxArray* arr) : ptr(mxArrayToUTF8String(arr)) {
        if (!ptr)
            mexErrMsgIdAndTxt("load_bl_tif:BadString", "Failed to convert string from mxArray");
    }
    MatlabString(const MatlabString&) = delete;
    MatlabString& operator=(const MatlabString&) = delete;
    MatlabString(MatlabString&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    MatlabString& operator=(MatlabString&& other) noexcept {
        if (this != &other) { mxFree(ptr); ptr = other.ptr; other.ptr = nullptr; }
        return *this;
    }
    ~MatlabString() { mxFree(ptr); }
    const char* get() const { return ptr; }
    operator const char*() const { return ptr; }
};

struct LoadTask {
    uint32_t in_row0, in_col0, crop_height, crop_width;
    uint32_t roi_height, roi_width, z_index;
    size_t out_row0, out_col0, pixels_per_slice;
    std::string file_path;
    bool transpose;
    LoadTask(uint32_t y0, uint32_t x0, size_t oy, size_t ox, uint32_t h, uint32_t w,
             uint32_t roi_h, uint32_t roi_w, uint32_t z, size_t pps, std::string fname, bool tr)
        : in_row0(y0), in_col0(x0), crop_height(h), crop_width(w), roi_height(roi_h),
          roi_width(roi_w), z_index(z), out_row0(oy), out_col0(ox), pixels_per_slice(pps),
          file_path(std::move(fname)), transpose(tr) {}
};

// Computes linear index into output array for given (row, col, slice).
// - If transpose==false: output is YXZ, i.e., [height, width, Z] (MATLAB column-major).
// - If transpose==true:  output is XYZ, i.e., [width, height, Z] (transpose per slice).
inline size_t compute_output_index(const LoadTask& task, uint32_t row, uint32_t col) noexcept {
    size_t r = task.out_row0 + row;
    size_t c = task.out_col0 + col;
    size_t slice = task.z_index;
    if (!task.transpose)
        return r + c * task.roi_height + slice * task.pixels_per_slice;  // [Y X Z]
    else
        return c + r * task.roi_width + slice * task.pixels_per_slice;   // [X Y Z]
}

struct TiffCloser { void operator()(TIFF* tif) const { if (tif) TIFFClose(tif); } };
using TiffHandle = std::unique_ptr<TIFF, TiffCloser>;

// Helper: parse all user inputs, validate, and return as structured arguments.
struct InputArgs {
    std::vector<std::string> file_paths;
    uint32_t roi_y0, roi_x0, roi_height, roi_width;
    bool transpose;
};
InputArgs parse_and_validate_inputs(int nrhs, const mxArray* prhs[]) {
    if (nrhs < 5 || nrhs > 6)
        mexErrMsgIdAndTxt("load_bl_tif:usage", "Usage: img = load_bl_tif(files, y, x, height, width[, transposeFlag])");
    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("load_bl_tif:args", "First argument must be a cell array of filenames");
    size_t num_slices = mxGetNumberOfElements(prhs[0]);
    std::vector<std::string> file_paths(num_slices);
    for (size_t i = 0; i < num_slices; ++i) {
        const mxArray* cell = mxGetCell(prhs[0], i);
        if (!mxIsChar(cell))
            mexErrMsgIdAndTxt("load_bl_tif:args", "File list must contain only strings.");
        MatlabString mstr(cell);
        if (!mstr.get() || !*mstr.get())
            mexErrMsgIdAndTxt("load_bl_tif:args", "Empty filename in cell %zu.", i + 1);
        file_paths[i] = mstr.get();
    }
    for (int i = 1; i <= 4; ++i) {
        if (!mxIsDouble(prhs[i]) || mxIsComplex(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1)
            mexErrMsgIdAndTxt("load_bl_tif:args", "Input argument %d must be a real double scalar.", i+1);
    }
    double y_in = mxGetScalar(prhs[1]);
    double x_in = mxGetScalar(prhs[2]);
    double h_in = mxGetScalar(prhs[3]);
    double w_in = mxGetScalar(prhs[4]);
    if (!mxIsFinite(y_in) || !mxIsFinite(x_in) || !mxIsFinite(h_in) || !mxIsFinite(w_in) ||
        y_in < 1 || x_in < 1 || h_in < 1 || w_in < 1)
        mexErrMsgIdAndTxt("load_bl_tif:args", "y, x, height, width must be finite positive numbers (1-based).");
    bool transpose = false;
    if (nrhs == 6) {
        const mxArray* flag = prhs[5];
        if (mxIsLogicalScalar(flag))
            transpose = mxIsLogicalScalarTrue(flag);
        else if ((mxIsInt32(flag) || mxIsUint32(flag)) && mxGetNumberOfElements(flag) == 1)
            transpose = (*static_cast<uint32_t*>(mxGetData(flag)) != 0);
        else
            mexErrMsgIdAndTxt("load_bl_tif:args", "transposeFlag must be logical or int32/uint32 scalar.");
    }
    return {file_paths,
            static_cast<uint32_t>(y_in - 1),
            static_cast<uint32_t>(x_in - 1),
            static_cast<uint32_t>(h_in),
            static_cast<uint32_t>(w_in),
            transpose};
}

// Helper: check all TIFF files for size, bit depth, ROI validity, etc.
uint16_t check_tiff_and_roi(const std::vector<std::string>& file_paths, uint32_t roi_y0, uint32_t roi_x0, uint32_t roi_h, uint32_t roi_w) {
    uint32_t img_width = 0, img_height = 0;
    uint16_t bits_per_sample = 0, global_bits_per_sample = 0, samples_per_pixel = 1;
    for (size_t z = 0; z < file_paths.size(); ++z) {
        TiffHandle tif(TIFFOpen(file_paths[z].c_str(), "r"));
        if (!tif)
            mexErrMsgIdAndTxt("load_bl_tif:file", "Cannot open file %s (slice %zu)", file_paths[z].c_str(), z+1);
        TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH , &img_width);
        TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &img_height);
        TIFFGetField(tif.get(), TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);
        if (z == 0) global_bits_per_sample = bits_per_sample;
        else if (bits_per_sample != global_bits_per_sample)
            mexErrMsgIdAndTxt("load_bl_tif:file", "Inconsistent bitsPerSample across slices.");
        if (samples_per_pixel != 1 ||
            (global_bits_per_sample != kSupportedBitDepth8 && global_bits_per_sample != kSupportedBitDepth16))
            mexErrMsgIdAndTxt("load_bl_tif:file", "Only 8/16-bit grayscale TIFFs (1 sample per pixel) supported.");
        if (roi_y0 + roi_h > img_height || roi_x0 + roi_w > img_width)
            mexErrMsgIdAndTxt("load_bl_tif:file", "ROI out of bounds for slice %zu (file: %s)", z+1, file_paths[z].c_str());
    }
    return global_bits_per_sample;
}

// Helper: create MATLAB output and return raw pointer
void* allocate_output(mxArray*& plhs0, mxClassID out_type, size_t out_h, size_t out_w, size_t num_slices) {
    size_t dims[3] = { out_h, out_w, num_slices };
    plhs0 = mxCreateNumericArray(3, dims, out_type, mxREAL);
    if (!plhs0)
        mexErrMsgIdAndTxt("load_bl_tif:alloc", "Failed to allocate output array.");
    return mxGetData(plhs0);
}

// Helper: Create per-slice tasks
std::vector<LoadTask> make_load_tasks(const std::vector<std::string>& file_paths, uint32_t roi_y0, uint32_t roi_x0, uint32_t roi_h, uint32_t roi_w, size_t pixels_per_slice, bool transpose) {
    std::vector<LoadTask> tasks;
    tasks.reserve(file_paths.size());
    for (size_t z = 0; z < file_paths.size(); ++z)
        tasks.emplace_back(roi_y0, roi_x0, 0, 0, roi_h, roi_w, roi_h, roi_w, static_cast<uint32_t>(z), pixels_per_slice, file_paths[z], transpose);
    return tasks;
}

// Helper: Safely read subregion from TIFF file to buffer (tiles or strips)
void decode_subregion_to_buffer(const LoadTask& task, TIFF* tif, uint8_t bytes_per_pixel,
                                std::vector<uint8_t>& block_buffer, std::vector<uint8_t>& temp_buffer) {
    uint32_t img_width = 0, img_height = 0;
    if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &img_width) ||
        !TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &img_height))
        throw std::runtime_error("Missing TIFFTAG_IMAGEWIDTH or IMAGELENGTH in file: " + task.file_path);

    uint16_t bits_per_sample = 0, samples_per_pixel = 1;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);

    if (samples_per_pixel != 1 ||
        (bits_per_sample != kSupportedBitDepth8 && bits_per_sample != kSupportedBitDepth16))
        throw std::runtime_error("Unsupported TIFF format: only 8/16-bit grayscale, 1 sample/pixel in file: " + task.file_path);

    const bool is_tiled = TIFFIsTiled(tif);

    if (is_tiled) {
        uint32_t tile_w = 0, tile_h = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH , &tile_w);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tile_h);
        if (tile_w == 0 || tile_h == 0)
            throw std::runtime_error("Invalid tile size in TIFF metadata in file: " + task.file_path);

        size_t tile_bytes = static_cast<size_t>(tile_w) * tile_h * bytes_per_pixel;
        if (tile_bytes > static_cast<size_t>(std::numeric_limits<tsize_t>::max()))
            throw std::runtime_error("Tile buffer too large (overflow risk)");
        if (tile_bytes > temp_buffer.size())
            temp_buffer.resize(tile_bytes);

        uint32_t prev_tile = UINT32_MAX;
        for (uint32_t row = 0; row < task.crop_height; ++row) {
            uint32_t img_y = task.in_row0 + row;
            for (uint32_t col = 0; col < task.crop_width; ++col) {
                uint32_t img_x = task.in_col0 + col;
                uint32_t tile_idx = TIFFComputeTile(tif, img_x, img_y, 0, 0);

                if (tile_idx != prev_tile) {
                    tsize_t ret = TIFFReadEncodedTile(
                        tif, tile_idx, temp_buffer.data(), tile_bytes
                    );
                    if (ret < 0)
                        throw std::runtime_error("TIFFReadEncodedTile failed (tile " + std::to_string(tile_idx) + ") in file: " + task.file_path);
                    prev_tile = tile_idx;
                }
                uint32_t rel_y = img_y % tile_h;
                uint32_t rel_x = img_x % tile_w;
                size_t   src_off = (static_cast<size_t>(rel_y) * tile_w + rel_x) * bytes_per_pixel;
                size_t   dst_off = (static_cast<size_t>(row) * task.crop_width + col) * bytes_per_pixel;
                std::memcpy(block_buffer.data() + dst_off, temp_buffer.data() + src_off, bytes_per_pixel);
            }
        }
    } else {
        uint32_t rows_per_strip = 0;
        TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
        if (rows_per_strip == 0) rows_per_strip = img_height;
        size_t strip_bytes = static_cast<size_t>(rows_per_strip) * img_width * bytes_per_pixel;
        if (strip_bytes > static_cast<size_t>(std::numeric_limits<tsize_t>::max()))
            throw std::runtime_error("Tile buffer too large (overflow risk)");
        if (strip_bytes > temp_buffer.size())
            temp_buffer.resize(strip_bytes);

        tstrip_t current_strip = (tstrip_t)-1;
        tsize_t  nbytes = 0;

        for (uint32_t row = 0; row < task.crop_height; ++row) {
            uint32_t tif_row   = task.in_row0 + row;
            tstrip_t strip_idx = TIFFComputeStrip(tif, tif_row, 0);
            if (strip_idx != current_strip) {
                nbytes = TIFFReadEncodedStrip(tif, strip_idx, temp_buffer.data(), strip_bytes);
                if (nbytes < 0)
                    throw std::runtime_error("TIFFReadEncodedStrip failed (strip " + std::to_string(strip_idx) + ") in file: " + task.file_path);
                current_strip = strip_idx;
            }
            const uint32_t rows_in_this_strip = static_cast<uint32_t>(nbytes / (img_width * bytes_per_pixel));
            uint32_t strip_start_row = strip_idx * rows_per_strip;
            uint32_t rel_row = tif_row - strip_start_row;
            if (rel_row >= rows_in_this_strip)
                throw std::runtime_error("Row " + std::to_string(tif_row+1) + " exceeds decoded strip size in file: " + task.file_path);
            uint8_t* scanline_ptr = temp_buffer.data() + (static_cast<size_t>(rel_row) * img_width * bytes_per_pixel);
            for (uint32_t col = 0; col < task.crop_width; ++col) {
                size_t src_off = (static_cast<size_t>(task.in_col0 + col)) * bytes_per_pixel;
                size_t dst_off = (static_cast<size_t>(row) * task.crop_width + col) * bytes_per_pixel;
                if (src_off + bytes_per_pixel > static_cast<size_t>(nbytes))
                    throw std::runtime_error("Column " + std::to_string(col+1) + " exceeds decoded strip size in file: " + task.file_path);
                std::memcpy(block_buffer.data() + dst_off, scanline_ptr + src_off, bytes_per_pixel);
            }
        }
    }
}

// Parallel producer/consumer decode and copy into output array.
void parallel_decode_and_copy(const std::vector<LoadTask>& tasks, void* out_data, size_t bytes_per_pixel) {
    const size_t num_slices = tasks.size();
    const size_t thread_pair_count = std::min(num_slices, get_available_cores());
    const size_t num_wires = thread_pair_count / kWires + ((thread_pair_count % kWires) ? 1 : 0);

    using TaskPtr = std::shared_ptr<std::vector<uint8_t>>;
    std::vector<std::unique_ptr<BoundedQueue<std::pair<size_t, TaskPtr>>>> queues_for_wires;
    queues_for_wires.reserve(num_wires);
    for (size_t w = 0; w < num_wires; ++w)
        queues_for_wires.emplace_back(std::make_unique<BoundedQueue<std::pair<size_t, TaskPtr>>>(2 * kWires));

    std::vector<std::thread> producer_threads, consumer_threads;
    producer_threads.reserve(thread_pair_count);
    consumer_threads.reserve(thread_pair_count);

    std::atomic<uint32_t> next_slice_index{0};
    std::vector<std::string> runtime_errors;
    std::mutex error_mutex;
    std::atomic<bool> abort_flag{false};
    auto thread_pairs = assign_thread_affinity_pairs(thread_pair_count);

    // Producer: decode each slice into buffer
    for (size_t t = 0; t < thread_pair_count; ++t) {
        BoundedQueue<std::pair<size_t, TaskPtr>>& queue_for_pair = *queues_for_wires[t / kWires];
        producer_threads.emplace_back([&, t] {
            set_thread_affinity(thread_pairs[t].producerLogicalCore);
            std::vector<uint8_t> temp_buffer;
            while (true) {
                if (abort_flag.load(std::memory_order_acquire)) break;
                uint32_t idx = next_slice_index.fetch_add(1, std::memory_order_relaxed);
                if (idx >= num_slices) break;
                const auto& task = tasks[idx];
                try {
                    TiffHandle tif(TIFFOpen(task.file_path.c_str(), "r"));
                    if (!tif) {
                        std::lock_guard<std::mutex> lck(error_mutex);
                        runtime_errors.emplace_back("Slice " + std::to_string(task.z_index + 1) + ": Cannot open file " + task.file_path);
                        abort_flag.store(true, std::memory_order_release);
                        break;
                    }
                    auto res = std::make_shared<std::vector<uint8_t>>(task.crop_height * task.crop_width * bytes_per_pixel);
                    decode_subregion_to_buffer(task, tif.get(), bytes_per_pixel, *res, temp_buffer);
                    queue_for_pair.push(std::make_pair(idx, res));
                } catch (const std::exception& ex) {
                    std::lock_guard<std::mutex> lck(error_mutex);
                    runtime_errors.emplace_back("Slice " + std::to_string(task.z_index + 1) + ": " + ex.what());
                    abort_flag.store(true, std::memory_order_release);
                    break;
                } catch (...) {
                    std::lock_guard<std::mutex> lck(error_mutex);
                    runtime_errors.emplace_back("Slice " + std::to_string(task.z_index + 1) + ": Unknown exception");
                    abort_flag.store(true, std::memory_order_release);
                    break;
                }
            }
            queue_for_pair.push({std::numeric_limits<size_t>::max(), nullptr}); // signal end
        });
    }
    // Consumer: copy to output, handle transpose vs. non-transpose
    for (size_t t = 0; t < thread_pair_count; ++t) {
        BoundedQueue<std::pair<size_t, TaskPtr>>& queue_for_pair = *queues_for_wires[t / kWires];
        consumer_threads.emplace_back([&, t] {
            set_thread_affinity(thread_pairs[t].consumerLogicalCore);
            while (true) {
                if (abort_flag.load(std::memory_order_acquire)) break;
                std::pair<size_t, TaskPtr> item;
                queue_for_pair.wait_and_pop(item);
                if (!item.second) break;
                size_t idx = item.first;
                const auto& task = tasks[idx];
                const auto& block_buffer = *item.second;
                // If not transposed, bulk copy (YXZ layout: [height, width, Z])
                if (!task.transpose) {
                    size_t dst_byte = task.z_index * task.pixels_per_slice * bytes_per_pixel;
                    size_t slice_bytes = task.pixels_per_slice * bytes_per_pixel;
                    std::memcpy(static_cast<uint8_t*>(out_data) + dst_byte, block_buffer.data(), slice_bytes);
                } else {
                    // If transposed, copy elementwise ([X, Y, Z])
                    for (uint32_t row = 0; row < task.crop_height; ++row) {
                        for (uint32_t col = 0; col < task.crop_width; ++col) {
                            size_t dst_elem = compute_output_index(task, row, col);
                            size_t dst_byte = dst_elem * bytes_per_pixel;
                            size_t src_byte = (row * task.crop_width + col) * bytes_per_pixel;
                            std::memcpy(static_cast<uint8_t*>(out_data) + dst_byte,
                                        block_buffer.data() + src_byte, bytes_per_pixel);
                        }
                    }
                }
            }
        });
    }
    for (auto& p : producer_threads) p.join();
    for (auto& c : consumer_threads) c.join();
    if (!runtime_errors.empty()) {
        std::ostringstream allerr;
        allerr << "Errors during load_bl_tif (producer/consumer):\n";
        for (const auto& s : runtime_errors) allerr << "  - " << s << "\n";
        mexErrMsgIdAndTxt("load_bl_tif:Error", "%s", allerr.str().c_str());
    }
}

// ==============================
//       ENTRY POINT
// ==============================
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    ensure_hwloc_initialized();
    try {
        InputArgs args = parse_and_validate_inputs(nrhs, prhs);
        uint16_t bits_per_sample = check_tiff_and_roi(args.file_paths, args.roi_y0, args.roi_x0, args.roi_height, args.roi_width);
        const mxClassID out_type = (bits_per_sample == 8) ? mxUINT8_CLASS : mxUINT16_CLASS;
        const uint8_t bytes_per_pixel = (bits_per_sample == 16) ? 2 : 1;
        size_t out_h = args.transpose ? args.roi_width  : args.roi_height;
        size_t out_w = args.transpose ? args.roi_height : args.roi_width;
        size_t pixels_per_slice = out_h * out_w;
        if (pixels_per_slice > kMaxPixelsPerSlice)
            mexErrMsgIdAndTxt("load_bl_tif:Error", "Requested ROI too large (>2^31 elements).");
        void* out_data = allocate_output(plhs[0], out_type, out_h, out_w, args.file_paths.size());
        auto tasks = make_load_tasks(args.file_paths, args.roi_y0, args.roi_x0, args.roi_height, args.roi_width, pixels_per_slice, args.transpose);
        parallel_decode_and_copy(tasks, out_data, bytes_per_pixel);
    } catch (const std::exception& ex) {
        mexErrMsgIdAndTxt("load_bl_tif:Error", "%s", ex.what());
    }
}
