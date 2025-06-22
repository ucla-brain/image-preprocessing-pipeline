/*==============================================================================
  save_bl_tif.cpp (experimental‑opt)
  ------------------------------------------------------------------------------
  High‑throughput Z‑slice saver for 3‑D MATLAB arrays (one TIFF per slice).

  VERSION  : 2025‑06‑21‑exp (thread‑local TIFF reuse, chunked dispatch,
                            NUMA‑local hugepages, raw gather‑write path)
  AUTHOR   : Keivan Moradi  (with ChatGPT‑o assistance)
  LICENSE  : GNU GPL v3   <https://www.gnu.org/licenses/>
==============================================================================*/

#include "mex.h"
#include "tiffio.h"

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include <unordered_map>

#if defined(__linux__)
#  include <fcntl.h>
#  include <numa.h>
#  include <sched.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <sys/uio.h>
#  include <unistd.h>
#endif

#if defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>
#  include <io.h>
#  include <fileapi.h>
#  include <fcntl.h>
#endif

/* ───────────────────────────── CONFIGURATION ────────────────────────────── */
static constexpr size_t kAlign          = 64;      // 64‑byte cache‑line alignment
static constexpr size_t kHugePageBytes  = 2 << 20; // 2 MiB
static constexpr size_t kChunkSz        = 8;       // slices claimed per atomic op

/* ───────────────────────────── TASK DESCRIPTION ─────────────────────────── */
struct SaveTask {
    const uint8_t* basePtr;
    size_t         sliceOffset;
    mwSize         rows, cols;      // dims AFTER transpose logic
    std::string    filePath;
    bool           alreadyXYZ;
    mxClassID      classId;
    uint16_t       compressionTag;
    size_t         bytesPerSlice;
    size_t         bytesPerPixel;
    size_t         sliceIndex;
};

/* ───────────────────────────── THREAD‑LOCAL SCRATCH ─────────────────────── */
static thread_local uint8_t* scratch_aligned   = nullptr;
static thread_local size_t   scratch_capacity  = 0;
static thread_local bool     scratch_hugepage  = false;

static void release_scratch()
{
#if defined(__linux__)
    if (scratch_aligned) {
        if (scratch_hugepage)
            munmap(scratch_aligned, scratch_capacity);
        else
            std::free(scratch_aligned);
    }
#else
    if (scratch_aligned)
        std::free(scratch_aligned);
#endif
    scratch_aligned  = nullptr;
    scratch_capacity = 0;
    scratch_hugepage = false;
}

static void* alloc_on_node(size_t bytes, int node, bool huge)
{
#if defined(__linux__)
    if (huge) {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p == MAP_FAILED) return nullptr;
        unsigned long nodemask = 1UL << node;
        mbind(p, bytes, MPOL_BIND, &nodemask, sizeof(nodemask)*8, 0);
        return p;
    }
    /* small buffers or huge failed → fallback */
    void* p = numa_alloc_onnode(bytes, node);
    return p;
#else
    (void)node; (void)huge;
    return nullptr;
#endif
}

static void ensure_scratch(size_t bytes)
{
    if (scratch_capacity >= bytes && scratch_aligned) return;
    release_scratch();

    /* round up to alignment */
    size_t aligned_bytes = ((bytes + kAlign - 1) / kAlign) * kAlign;

#if defined(__linux__)
    int node = 0;
    if (numa_available() != -1)
        node = numa_node_of_cpu(sched_getcpu());

    bool huge = (aligned_bytes >= kHugePageBytes);
    if (void* p = alloc_on_node(aligned_bytes, node, huge)) {
        scratch_aligned  = static_cast<uint8_t*>(p);
        scratch_capacity = aligned_bytes;
        scratch_hugepage = huge;
        return;
    }
#endif
    /* portable fallback */
#if defined(__cpp_aligned_new)
    scratch_aligned = static_cast<uint8_t*>(::operator new(aligned_bytes, std::align_val_t{kAlign}));
#else
    if (posix_memalign(reinterpret_cast<void**>(&scratch_aligned), kAlign, aligned_bytes) != 0)
        throw std::bad_alloc();
#endif
    scratch_capacity = aligned_bytes;
    scratch_hugepage = false;
}

/* ───────────────────────── RAW GATHER‑WRITE (UNCOMPRESSED) ──────────────── */
static void write_tiff_raw(const SaveTask& t, const uint8_t* buf)
{
    /* Build minimal little‑endian classic TIFF header */
    const uint32_t width  = static_cast<uint32_t>(t.cols);
    const uint32_t height = static_cast<uint32_t>(t.rows);
    const uint32_t bcount = static_cast<uint32_t>(t.bytesPerSlice);
    const uint16_t bps    = (t.bytesPerPixel == 2) ? 16 : 8;

    /* Offsets: header(8) + dirCount(2) + 12*N + nextIFD(4) = 8+2+120+4 = 134 */
    constexpr uint16_t N = 10;
    const uint32_t ifdOffset = 8;
    const uint32_t pixelOffset = 134;

    std::array<uint8_t, 8> hdr{};
    hdr[0] = 'I'; hdr[1] = 'I';
    hdr[2] = 42;  hdr[3] = 0;
    std::memcpy(&hdr[4], &ifdOffset, 4);

    std::vector<uint8_t> ifd(2 + N*12 + 4, 0);
    uint16_t* dirCount = reinterpret_cast<uint16_t*>(ifd.data());
    *dirCount = N;

    auto putEntry = [&](size_t idx, uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
        uint8_t* base = &ifd[2 + idx*12];
        std::memcpy(base+0, &tag,   2);
        std::memcpy(base+2, &type,  2);
        std::memcpy(base+4, &count, 4);
        std::memcpy(base+8, &value, 4);
    };

    putEntry(0, 256, 4, 1, width);          // ImageWidth  LONG
    putEntry(1, 257, 4, 1, height);         // ImageLength LONG
    putEntry(2, 258, 3, 1, bps);            // BitsPerSample SHORT (fits in value)
    putEntry(3, 259, 3, 1, 1);              // Compression = 1 (none)
    putEntry(4, 262, 3, 1, 1);              // Photometric = MINISBLACK
    putEntry(5, 273, 4, 1, pixelOffset);    // StripOffsets
    putEntry(6, 277, 3, 1, 1);              // SamplesPerPixel
    putEntry(7, 278, 4, 1, height);         // RowsPerStrip = image height
    putEntry(8, 279, 4, 1, bcount);         // StripByteCounts
    putEntry(9, 284, 3, 1, 1);              // PlanarConfig = CONTIG

    /* next IFD = 0 (end) already zero‑initialised */

    const std::string tmpPath = t.filePath + ".tmp";

#if defined(_WIN32)
    HANDLE h = CreateFileA(tmpPath.c_str(), GENERIC_WRITE, 0, nullptr,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE)
        throw std::runtime_error("Cannot open " + tmpPath);
    DWORD written = 0;
    if (!WriteFile(h, hdr.data(), static_cast<DWORD>(hdr.size()), &written, nullptr) || written != hdr.size())
        goto write_fail;
    if (!WriteFile(h, ifd.data(), static_cast<DWORD>(ifd.size()), &written, nullptr) || written != ifd.size())
        goto write_fail;
    if (!WriteFile(h, buf, static_cast<DWORD>(bcount), &written, nullptr) || written != bcount)
        goto write_fail;
    FlushFileBuffers(h);
    CloseHandle(h);
    if (!MoveFileExA(tmpPath.c_str(), t.filePath.c_str(), MOVEFILE_REPLACE_EXISTING))
        throw std::runtime_error("Rename failed: " + tmpPath);
    return;
write_fail:
    CloseHandle(h);
    std::remove(tmpPath.c_str());
    throw std::runtime_error("WriteFile failed on " + tmpPath);
#else   /* POSIX */
    int fd = ::open(tmpPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0)
        throw std::runtime_error("Cannot open " + tmpPath + ": " + std::strerror(errno));

    auto xwrite = [&](const void* p, size_t n) {
        const char* c = static_cast<const char*>(p);
        while (n) {
            ssize_t w = ::write(fd, c, n);
            if (w <= 0) {
                ::close(fd);
                std::remove(tmpPath.c_str());
                throw std::runtime_error("Write failed on " + tmpPath);
            }
            c += w; n -= w;
        }
    };

    xwrite(hdr.data(), hdr.size());
    xwrite(ifd.data(), ifd.size());
    xwrite(buf, bcount);
    ::fsync(fd);
    ::close(fd);
    if (::rename(tmpPath.c_str(), t.filePath.c_str()) != 0) {
        std::remove(tmpPath.c_str());
        throw std::runtime_error("Rename failed: " + tmpPath);
    }
#endif
}

/* ───────────────────────────── TIFF SLICE WRITE ─────────────────────────── */
static void save_slice(const SaveTask& t)
{
    const mwSize srcRows = t.alreadyXYZ ? t.cols : t.rows;
    const mwSize srcCols = t.alreadyXYZ ? t.rows : t.cols;

    const uint8_t* src = t.basePtr + t.sliceOffset;
    const bool needTranspose = !t.alreadyXYZ;
    const bool rawPath       = (t.compressionTag == COMPRESSION_NONE);

    /* ── Direct raw write if no compression ─────────────────────────────── */
    if (rawPath) {
        const uint8_t* ioBuf = nullptr;
        if (!needTranspose) {
            ioBuf = src;          // Already in [X Y] order
        } else {
            /* Make transposed copy */
            ensure_scratch(t.bytesPerSlice);
            uint8_t* dst = scratch_aligned;
            for (mwSize col = 0; col < srcCols; ++col) {
                const uint8_t* srcColumn = src + col * t.rows * t.bytesPerPixel;
                for (mwSize row = 0; row < srcRows; ++row) {
                    size_t dstIdx = (static_cast<size_t>(row) * srcCols + col) * t.bytesPerPixel;
                    std::memcpy(dst + dstIdx,
                                srcColumn + row * t.bytesPerPixel,
                                t.bytesPerPixel);
                }
            }
            ioBuf = dst;
        }
        write_tiff_raw(t, ioBuf);
        return;
    }

    /* ── Compressed path via LibTIFF (thread‑local handle reuse) ─────────── */
    thread_local struct {
        TIFF*       tif        = nullptr;
        std::string tmpPath;
    } tls;

    /* create tmp handle on first use */
    if (!tls.tif) {
        tls.tmpPath = []() {
#if defined(_WIN32)
            char buf[MAX_PATH];
            GetTempFileNameA(".", "sbt", 0, buf);
            return std::string(buf);
#else
            char tmpl[] = "/tmp/save_bl_tifXXXXXX";
            int fd = mkstemp(tmpl);
            if (fd >= 0) close(fd);
            return std::string(tmpl);
#endif
        }();
        tls.tif = TIFFOpen(tls.tmpPath.c_str(), "w+");
        if (!tls.tif)
            throw std::runtime_error("Cannot open scratch TIFF " + tls.tmpPath);
    }

    /* reset file & directory */
#if defined(__linux__)
    {
        int fd = TIFFFileno(tls.tif);
        ::ftruncate(fd, 0);
        ::lseek(fd, 0, SEEK_SET);
    }
#elif defined(_WIN32)
    {
        HANDLE h = (HANDLE)_get_osfhandle(TIFFFileno(tls.tif));
        SetFilePointer(h, 0, nullptr, FILE_BEGIN);
        SetEndOfFile(h);
    }
#endif
    TIFFRewriteDirectory(tls.tif);

    /* write tags */
    TIFFSetField(tls.tif, TIFFTAG_IMAGEWIDTH,  srcCols);
    TIFFSetField(tls.tif, TIFFTAG_IMAGELENGTH, srcRows);
    TIFFSetField(tls.tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tls.tif, TIFFTAG_BITSPERSAMPLE, t.bytesPerPixel == 2 ? 16 : 8);
    TIFFSetField(tls.tif, TIFFTAG_COMPRESSION, t.compressionTag);
    TIFFSetField(tls.tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tls.tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tls.tif, TIFFTAG_ROWSPERSTRIP, srcRows);

    /* prepare buffer (transpose if necessary) */
    ensure_scratch(t.bytesPerSlice);
    uint8_t* writeBuf = scratch_aligned;

    if (!needTranspose) {
        std::memcpy(writeBuf, src, t.bytesPerSlice);
    } else {
        for (mwSize col = 0; col < srcCols; ++col) {
            const uint8_t* srcColumn = src + col * t.rows * t.bytesPerPixel;
            for (mwSize row = 0; row < srcRows; ++row) {
                size_t dstIdx = (static_cast<size_t>(row) * srcCols + col) * t.bytesPerPixel;
                std::memcpy(writeBuf + dstIdx,
                            srcColumn + row * t.bytesPerPixel,
                            t.bytesPerPixel);
            }
        }
    }

    tsize_t nWritten = TIFFWriteEncodedStrip(tls.tif, 0, writeBuf,
                                             static_cast<tsize_t>(t.bytesPerSlice));
    if (nWritten < 0)
        throw std::runtime_error("TIFFWriteEncodedStrip failed on slice " + std::to_string(t.sliceIndex));

    TIFFFlush(tls.tif);

    /* atomically rename finished file */
#if defined(_WIN32)
    fflush(nullptr);
    if (!MoveFileExA(tls.tmpPath.c_str(), t.filePath.c_str(), MOVEFILE_REPLACE_EXISTING))
        throw std::runtime_error("MoveFileEx failed on slice «" + std::to_string(t.sliceIndex) + "»");
#else
    if (::rename(tls.tmpPath.c_str(), t.filePath.c_str()) != 0)
        throw std::runtime_error("rename failed on slice «" + std::to_string(t.sliceIndex) + "»: " + std::strerror(errno));
#endif
}

/* ───────────────────────────── FILE LIST CACHE ──────────────────────────── */
struct FileListCacheKey {
    const void* mxArrayPtr;
    size_t      length;
    bool operator==(const FileListCacheKey& o) const {
        return mxArrayPtr == o.mxArrayPtr && length == o.length;
    }
};
namespace std {
    template<>
    struct hash<FileListCacheKey> {
        size_t operator()(const FileListCacheKey& k) const {
            return std::hash<const void*>()(k.mxArrayPtr) ^ std::hash<size_t>()(k.length);
        }
    };
}

/* ───────────────────────────── ONE‑SHOT CONTEXT ─────────────────────────── */
struct CallContext {
    std::shared_ptr<const std::vector<SaveTask>> tasks;
    std::atomic_size_t nextIndex{0};
    std::mutex errMutex;
    std::vector<std::string> errors;
};

static void worker_entry(CallContext& ctx, int tid)
{
#if defined(__linux__)
    if (numa_available() != -1) {
        int max_node = numa_max_node();
        int target_node = tid % (max_node + 1);
        numa_run_on_node(target_node);
    }
    pthread_setname_np(pthread_self(), "save_bl_tif");
    sched_param param{}; pthread_setschedparam(pthread_self(), SCHED_BATCH, &param);
#endif

    const auto& jobs = *ctx.tasks;
    size_t base = ctx.nextIndex.fetch_add(kChunkSz, std::memory_order_relaxed);

    while (base < jobs.size()) {
        for (size_t idx = base; idx < base + kChunkSz && idx < jobs.size(); ++idx) {
            try {
                save_slice(jobs[idx]);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> g(ctx.errMutex);
                ctx.errors.emplace_back(e.what());
            }
        }
        base = ctx.nextIndex.fetch_add(kChunkSz, std::memory_order_relaxed);
    }

    release_scratch(); // free per‑thread buffer when done
}

/* ───────────────────────────── MEX ENTRY POINT ─────────────────────────── */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    try {
        if (nrhs < 4 || nrhs > 5)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                "Usage: save_bl_tif(volume, fileList, orderFlag, compression [, nThreads])");

        /* volume checks */
        const mxArray* V = prhs[0];
        if (!mxIsUint8(V) && !mxIsUint16(V))
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be uint8 or uint16");
        const mwSize* dims = mxGetDimensions(V);
        const size_t dim0 = dims[0], dim1 = dims[1];
        const size_t dim2 = (mxGetNumberOfDimensions(V) == 3) ? dims[2] : 1;

        const uint8_t* basePtr   = static_cast<const uint8_t*>(mxGetData(V));
        const mxClassID classId  = mxGetClassID(V);
        const size_t bytesPerPx  = (classId == mxUINT16_CLASS) ? 2 : 1;
        const size_t bytesPerSl  = dim0 * dim1 * bytesPerPx;

        bool alreadyXYZ = mxIsLogicalScalarTrue(prhs[2]) ||
                          (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0.0);

        char* cstr = mxArrayToUTF8String(prhs[3]);
        if (!cstr) mexErrMsgIdAndTxt("save_bl_tif:Input", "Invalid compression string");
        std::string compStr(cstr); mxFree(cstr);
        uint16_t compTag = COMPRESSION_NONE;
        if (compStr == "lzw") compTag = COMPRESSION_LZW;
        else if (compStr == "deflate" || compStr == "zip") compTag = COMPRESSION_DEFLATE;
        else if (compStr != "none")
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Compression must be 'none', 'lzw', or 'deflate'");

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != dim2)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList must match Z dimension");

        /* file list cache */
        static std::unordered_map<FileListCacheKey, std::vector<std::string>> fileCache;
        FileListCacheKey key{prhs[1], dim2};
        std::vector<std::string> paths;
        if (auto it = fileCache.find(key); it != fileCache.end()) paths = it->second;
        else {
            paths.resize(dim2);
            for (size_t z = 0; z < dim2; ++z) {
                mxArray* cell = mxGetCell(prhs[1], z);
                if (!mxIsChar(cell))
                    mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList elements must be strings");
                char* s = mxArrayToUTF8String(cell);
                paths[z] = s; mxFree(s);
            }
            fileCache[key] = paths;
        }

        /* build tasks */
        auto tasks = std::make_shared<std::vector<SaveTask>>();
        tasks->reserve(dim2);
        for (size_t z = 0; z < dim2; ++z)
            tasks->push_back({ basePtr, z*bytesPerSl, dim0, dim1,
                               paths[z], alreadyXYZ, classId,
                               compTag, bytesPerSl, bytesPerPx, z });

        /* libtiff warm‑up */
        TIFF* warm = TIFFOpen("/dev/null", "r"); if (warm) TIFFClose(warm);

        /* thread pool launch */
        CallContext ctx; ctx.tasks = tasks;

        size_t hw = std::thread::hardware_concurrency();
        if (!hw) hw = tasks->size();
        size_t maxThreads = std::min(hw, tasks->size());
        if (nrhs == 5) {
            double req = mxGetScalar(prhs[4]);
            if (!(req > 0)) mexErrMsgIdAndTxt("save_bl_tif:Input", "nThreads must be > 0");
            maxThreads = std::min<size_t>(static_cast<size_t>(req), tasks->size());
        }

        std::vector<std::thread> workers; workers.reserve(maxThreads);
        for (size_t i = 0; i < maxThreads; ++i)
            workers.emplace_back(worker_entry, std::ref(ctx), static_cast<int>(i));
        for (auto& t : workers) t.join();

        if (!ctx.errors.empty()) {
            std::string msg("save_bl_tif errors:\n");
            for (auto& e : ctx.errors) msg += "  - " + e + '\n';
            mexErrMsgIdAndTxt("save_bl_tif:Runtime", "%s", msg.c_str());
        }

        if (nlhs) plhs[0] = const_cast<mxArray*>(prhs[0]);
    }
    catch (const std::exception& e) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception", "%s", e.what());
    }
}
