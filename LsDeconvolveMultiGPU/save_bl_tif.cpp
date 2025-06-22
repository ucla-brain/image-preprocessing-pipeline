/*==============================================================================
  save_bl_tif.cpp  (experimental-opt patched-r2)
  ------------------------------------------------------------------------------
  High-throughput Z-slice saver for 3-D MATLAB arrays (one TIFF per slice).

  VERSION  : 2025-06-21-exp-r2  (thread-local TIFF reuse, chunked fetch,
                                NUMA-local hugepages, raw gather-write path)
  AUTHOR   : Keivan Moradi  (with ChatGPT-o assistance)
  LICENSE  : GNU GPL v3   <https://www.gnu.org/licenses/>
==============================================================================*/

#include "mex.h"
#include "tiffio.h"

#include <atomic>
#include <cerrno>     // errno
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <type_traits>

#if defined(__linux__)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sched.h>
#  include <numa.h>
#  include <sys/uio.h>
#  ifdef __has_include
#    if __has_include(<numaif.h>)
#      include <numaif.h>
#    endif
#  endif
#  if !defined(O_BINARY)
#    define O_BINARY 0
#  endif
#  include <unistd.h>
#  define ACCESS access
#endif

#if defined(_WIN32)
#  include <windows.h>
#  include <io.h>
#  include <io.h>     // _access
#  define ACCESS _access
#endif

// Uncomment to enable thread pinning for benchmarking
//#define PIN_THREADS

inline void guard_overwrite_writable(const std::string& path)
{
    /* Does the path exist?  (If it does not, we’re fine.) */
    if (ACCESS(path.c_str(), F_OK) == 0)           // file is present
    {
        /* Is it writable?  (W_OK asks the kernel, not ACLs on Windows.) */
        if (ACCESS(path.c_str(), W_OK) == -1)      // ❌ not writable
        {
#if defined(_WIN32)
            if (errno == EACCES || errno == EPERM)
#else
            if (errno == EACCES)
#endif
                throw std::runtime_error("Refused to overwrite read-only file: "
                                         + path);
            /* For other errno values (e.g. ETXTBSY, EPERM) you can decide
               whether to throw or fall through; EACCES is the canonical
               “read-only” case. */
        }
    }
}

/* ───────────────────────── Utility: NUMA-aware / hugepage alloc ─────────── */
namespace {

static void* alloc_on_node(size_t bytes, int node, bool wantHuge)
{
    void* p = nullptr;
#if defined(__linux__)
    // try hugepage first
    if (wantHuge && bytes >= (2UL << 20)) {
        size_t hugeSz = ((bytes + (2UL << 20) - 1) >> 21) << 21; // align 2 MiB
        void* hp = ::mmap(nullptr, hugeSz, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (hp != MAP_FAILED) {
#       if defined(MPOL_BIND)
            unsigned long nodemask = 1UL << node;
            ::mbind(hp, hugeSz, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, 0);
#       endif
            return hp;
        }
    }
    // fall back to libnuma-based alloc on node
    p = ::numa_alloc_onnode(bytes, node);
    if (p) return p;
#endif
    // generic aligned alloc
    constexpr size_t ALIGN = 64;
    if (::posix_memalign(&p, ALIGN, ((bytes + ALIGN - 1) / ALIGN) * ALIGN) != 0)
        p = nullptr;
    return p;
}

static void free_on_node(void* ptr, size_t bytes, bool huge)
{
#if defined(__linux__)
    if (!ptr) return;
    if (huge) { ::munmap(ptr, bytes); return; }
    ::numa_free(ptr, bytes);
#else
    (void)bytes; (void)huge; std::free(ptr);
#endif
}

} // unnamed namespace

/* ───────────────────────────── TASK DESCRIPTION ─────────────────────────── */
struct SaveTask {
    const uint8_t* basePtr;
    size_t         sliceOffset;
    mwSize         rows, cols;
    std::string    filePath;
    bool           alreadyXYZ;
    mxClassID      classId;
    uint16_t       compressionTag;
    size_t         bytesPerSlice;
    size_t         bytesPerPixel;
    size_t         sliceIndex;
};

/* ───────────── Thread-local scratch & TIFF handle (reuse between slices) ─── */
static thread_local uint8_t* scratch_buf   = nullptr;
static thread_local size_t   scratch_bytes = 0;
static thread_local bool     scratch_huge  = false;

struct TiffLocal {
    TIFF* tif        = nullptr;
    std::string tmpPath;
    void ensure_open(const std::string& path) {
        if (!tif) {
            tif = TIFFOpen(path.c_str(), "w");
            if (!tif) throw std::runtime_error("Cannot open tmp " + path);
            tmpPath = path;
        }
    }
    ~TiffLocal() { if (tif) TIFFClose(tif); }
};
static thread_local std::unique_ptr<TiffLocal> tl_tiff;

static void ensure_scratch(size_t want, int node)
{
    if (scratch_buf && scratch_bytes >= want) return;
    if (scratch_buf) free_on_node(scratch_buf, scratch_bytes, scratch_huge);

    void* p = alloc_on_node(want, node, /*wantHuge=*/true);
    scratch_huge  = (p && want >= (2UL<<20) &&
                     ((uintptr_t)p & ((2UL<<20)-1)) == 0);   // mmap+hugetlb?
    if (!p) throw std::bad_alloc();

    scratch_buf   = static_cast<uint8_t*>(p);
    scratch_bytes = want;
}

/* ─────────────────────────── Raw gather-write helpers ───────────────────── */
namespace rawtiff {
#pragma pack(push,1)
struct Header { uint16_t le; uint16_t magic; uint32_t ifdOff; };
#pragma pack(pop)
static constexpr uint8_t ifdBlob[] = {
    0x0E,0x00,
    0x00,0x01,0x00,0x04,0x00,0x00,0x00,0x01, 0,0,0,0,        // ImageWidth
    0x01,0x01,0x00,0x04,0x00,0x00,0x00,0x01, 0,0,0,0,        // ImageLength
    0x02,0x01,0x00,0x03,0x00,0x00,0x00,0x01, 0,0, 0,0,       // BitsPerSample
    0x03,0x01,0x00,0x03,0x00,0x00,0x00,0x01, 0x01,0x00,0,0,  // Compression (none)
    0x06,0x01,0x00,0x03,0x00,0x00,0x00,0x01, 0x01,0x00,0,0,  // Photometric (min-is-black)
    0x11,0x01,0x00,0x04,0x00,0x00,0x00,0x01, 0,0,0,0,        // StripOffsets
    0x15,0x01,0x00,0x03,0x00,0x00,0x00,0x01, 0x01,0x00,0,0,  // SamplesPerPixel
    0x16,0x01,0x00,0x04,0x00,0x00,0x00,0x01, 0,0,0,0,        // RowsPerStrip
    0x17,0x01,0x00,0x04,0x00,0x00,0x00,0x01, 0,0,0,0,        // StripByteCounts
    0x1A,0x01,0x00,0x05,0x00,0x00,0x00,0x01, 0,0,0,0,        // XResolution
    0x1B,0x01,0x00,0x05,0x00,0x00,0x00,0x01, 0,0,0,0,        // YResolution
    0x28,0x01,0x00,0x03,0x00,0x00,0x00,0x01, 0x02,0x00,0,0,  // ResolutionUnit (inch)
    0x31,0x01,0x00,0x02,0x00,0x00,0x00,0x01, 'M',0,0,0,      // Software
    0x00,0x00,0x00,0x00                                     // next IFD = 0
};
}

/* ───────────────────────────── LOW-LEVEL TIFF WRITE ─────────────────────── */
static void save_slice(const SaveTask& t, int node)
{
    /* ---- 1.  Refuse to clobber read-only files -------------------------- */
    guard_overwrite_writable(t.filePath);

    const mwSize srcRows = t.alreadyXYZ ? t.cols : t.rows;
    const mwSize srcCols = t.alreadyXYZ ? t.rows : t.cols;
    const uint8_t* src   = t.basePtr + t.sliceOffset;

    const bool direct = (t.compressionTag == COMPRESSION_NONE && t.alreadyXYZ);
    const std::string tmpPath = t.filePath + ".tmp";

/* ───────────── 2.  RAW gather-write path (uncompressed & already-XYZ) ─── */
    if (direct)
    {
        /* -------- build header + IFD in little-endian --------------------- */
        struct Entry { uint16_t tag,type; uint32_t count,val; };

        constexpr uint16_t TIFF_LONG = 4;
        constexpr uint16_t TIFF_SHORT= 3;

        Entry e[9] = {
            {256, TIFF_LONG , 1, uint32_t(srcCols)},          // ImageWidth
            {257, TIFF_LONG , 1, uint32_t(srcRows)},          // ImageLength
            {258, TIFF_SHORT, 1, uint32_t(t.bytesPerPixel==2?16:8)}, // BitsPerSample
            {259, TIFF_SHORT, 1, 1},                          // Compression=1 (none)
            {262, TIFF_SHORT, 1, 1},                          // Photometric=min-is-black
            {273, TIFF_LONG , 1, 0},                          // StripOffsets (fix later)
            {277, TIFF_SHORT, 1, 1},                          // SamplesPerPixel
            {278, TIFF_LONG , 1, uint32_t(srcRows)},          // RowsPerStrip
            {279, TIFF_LONG , 1, uint32_t(t.bytesPerSlice)}   // StripByteCounts
        };

        const uint16_t nEntries = sizeof(e)/sizeof(e[0]);
        const uint32_t ifdOffset = 8;                         // right after header
        const uint32_t ifdSize   = 2 + nEntries*12 + 4;       // count + entries + next=0
        const uint32_t pixelOffset = ifdOffset + ifdSize;     // data starts here
        e[5].val = pixelOffset;                               // patch StripOffsets

        /* ---- marshal little-endian header & IFD into stack buffers ------- */
        uint8_t hdr[8] = { 'I','I', 42,0, 8,0,0,0 };          // LE, magic 42, IFD @8
        uint8_t ifd[ifdSize];
        auto wr16=[&](uint8_t* p,uint16_t v){ p[0]=v&0xFF; p[1]=v>>8; };
        auto wr32=[&](uint8_t* p,uint32_t v){
            p[0]=v&0xFF; p[1]=v>>8; p[2]=v>>16; p[3]=v>>24; };
        wr16(ifd, nEntries);
        uint8_t* cur = ifd+2;
        for (uint16_t i=0;i<nEntries;++i){
            wr16(cur+0, e[i].tag);
            wr16(cur+2, e[i].type);
            wr32(cur+4, e[i].count);
            wr32(cur+8,  e[i].val);
            cur += 12;
        }
        std::memset(cur, 0, 4);     // next IFD offset = 0

        /* ---- write header + IFD + pixel data in a single system call ----- */
        int fd = ::open(tmpPath.c_str(),
                        O_CREAT|O_TRUNC|O_WRONLY|O_BINARY, 0644);
        if (fd == -1) throw std::runtime_error("open "+tmpPath);

#if defined(__linux__)
        struct iovec iov[3] = {
            {hdr, sizeof(hdr)},
            {ifd, sizeof(ifd)},
            {const_cast<uint8_t*>(src), t.bytesPerSlice}
        };
        if (::writev(fd, iov, 3) !=
            ssize_t(sizeof(hdr)+sizeof(ifd)+t.bytesPerSlice))
            throw std::runtime_error("writev failed on "+tmpPath);
#else
        if (::write(fd, hdr, sizeof(hdr)) != ssize_t(sizeof(hdr)) ||
            ::write(fd, ifd, sizeof(ifd)) != ssize_t(sizeof(ifd)) ||
            ::write(fd, src, t.bytesPerSlice)!=ssize_t(t.bytesPerSlice))
            throw std::runtime_error("write failed on "+tmpPath);
#endif
        ::close(fd);

        /* ---- atomic replace ------------------------------------------------*/
        ::unlink(t.filePath.c_str());                          // drop stub
        if (::rename(tmpPath.c_str(), t.filePath.c_str()) != 0)
            throw std::runtime_error("rename failed on "+tmpPath+
                                     " → "+t.filePath);
        return;
    }

/* ───────────── 3.  LibTIFF path (compressed or needs transpose) ────────── */
    if (!tl_tiff) tl_tiff = std::make_unique<TiffLocal>();
    tl_tiff->ensure_open(tmpPath);
    TIFF* tif = tl_tiff->tif;

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,       srcCols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH,      srcRows);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL,  1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,    (t.bytesPerPixel==2)?16:8);
    TIFFSetField(tif, TIFFTAG_COMPRESSION,      t.compressionTag);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,      PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG,     PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,     srcRows);

    /* -------- build / transpose I/O buffer -------------------------------- */
    const uint8_t* ioBuf = nullptr;
    if (t.alreadyXYZ && t.compressionTag==COMPRESSION_NONE){
        ioBuf = src;
    } else {
        ensure_scratch(t.bytesPerSlice, node);
        uint8_t* dst = scratch_buf;
        if (!t.alreadyXYZ){
            for (mwSize c=0;c<srcCols;++c){
                const uint8_t* colSrc = src + c*t.rows*t.bytesPerPixel;
                for (mwSize r=0;r<srcRows;++r){
                    size_t d = (size_t(r)*srcCols + c)*t.bytesPerPixel;
                    std::memcpy(dst+d, colSrc+r*t.bytesPerPixel, t.bytesPerPixel);
                }
            }
        } else {
            std::memcpy(dst, src, t.bytesPerSlice);
        }
        ioBuf = dst;
    }

    tsize_t nWritten =
        (t.compressionTag==COMPRESSION_NONE)
            ? TIFFWriteRawStrip    (tif,0,const_cast<uint8_t*>(ioBuf),
                                    (tsize_t)t.bytesPerSlice)
            : TIFFWriteEncodedStrip(tif,0,const_cast<uint8_t*>(ioBuf),
                                    (tsize_t)t.bytesPerSlice);
    if (nWritten < 0)
        throw std::runtime_error("TIFF write failed on slice "+
                                 std::to_string(t.sliceIndex));

    TIFFRewriteDirectory(tif);                         // commit IFD

    ::unlink(t.filePath.c_str());                      // drop stub
    if (::rename(tmpPath.c_str(), t.filePath.c_str()) != 0)
        throw std::runtime_error("rename failed on "+tmpPath+
                                 " → "+t.filePath);

    TIFFFlush(tif);
    TIFFClose(tif);                                    // fresh tmp next slice
    tl_tiff->tif = nullptr;
}


/* ─────────────────────────── Thread worker / dispatcher ─────────────────── */
struct CallContext {
    const std::vector<SaveTask>* tasks;
    std::atomic_size_t next{0};
    std::mutex errMu;
    std::vector<std::string> errs;
};

static void worker_entry(CallContext& ctx, int tid, int numaNode)
{
    constexpr size_t CHUNK = 8;
    auto jobList = ctx.tasks;
    for (;;) {
        size_t base = ctx.next.fetch_add(CHUNK, std::memory_order_relaxed);
        if (base >= jobList->size()) break;
        size_t end = std::min(base+CHUNK, jobList->size());
        for (size_t i=base; i<end; ++i) {
            try { save_slice((*jobList)[i], numaNode); }
            catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(ctx.errMu);
                ctx.errs.emplace_back(e.what());
            }
        }
    }
}

/* ──────────────────────────────── MEX ENTRY ─────────────────────────────── */
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    try {
        /* -------- argument check -------- */
        if (nrhs != 4 && nrhs != 5)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                "Usage: save_bl_tif(volume, fileList, orderXYZ, compression [, nThreads])");

        const mxArray* V = prhs[0];
        if (!mxIsUint8(V) && !mxIsUint16(V))
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be uint8/uint16.");

        const mwSize* dims = mxGetDimensions(V);
        const size_t R = dims[0], C = dims[1];
        const size_t Z = (mxGetNumberOfDimensions(V)==3) ? dims[2] : 1;

        bool alreadyXYZ = mxIsLogicalScalarTrue(prhs[2]) ||
                          (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2])!=0);
        char* cstr = mxArrayToUTF8String(prhs[3]);
        if (!cstr) mexErrMsgIdAndTxt("save_bl_tif:Input","Bad compression arg");
        std::string compStr(cstr); mxFree(cstr);
        uint16_t compTag = (compStr=="lzw")?COMPRESSION_LZW:
                           (compStr=="deflate"||compStr=="zip")?COMPRESSION_DEFLATE:
                           COMPRESSION_NONE;
        if (compTag==COMPRESSION_NONE && compStr!="none")
            mexErrMsgIdAndTxt("save_bl_tif:Input","Compression must be none/lzw/deflate");

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1])!=Z)
            mexErrMsgIdAndTxt("save_bl_tif:Input","fileList must match Z dim");

        /* -------- build path list (no cache for brevity) -------- */
        std::vector<std::string> paths(Z);
        for (size_t k=0;k<Z;++k){
            mxArray* s = mxGetCell(prhs[1],k);
            char* p = mxArrayToUTF8String(s);
            paths[k]=p; mxFree(p);
        }

        /* -------- build task array -------- */
        const uint8_t* base = static_cast<const uint8_t*>(mxGetData(V));
        mxClassID id = mxGetClassID(V);
        size_t bpp = (id==mxUINT16_CLASS)?2:1;
        size_t bpslice = R*C*bpp;

        std::vector<SaveTask> tasks;
        tasks.reserve(Z);
        for (size_t z=0; z<Z; ++z)
            tasks.emplace_back(SaveTask{ base, z*bpslice, R, C,
                                         paths[z], alreadyXYZ, id, compTag,
                                         bpslice, bpp, z });

        /* -------- prepare thread pool -------- */
        CallContext ctx; ctx.tasks = &tasks;

        size_t hw = std::thread::hardware_concurrency();
        if (hw==0) hw = 1;
        size_t nThreads = std::min(hw, tasks.size());
        if (nrhs==5){
            double d = mxGetScalar(prhs[4]);
            if (d>0) nThreads = std::min((size_t)d, tasks.size());
        }

        size_t numaNodes = 0;
#if defined(__linux__)
        if (numa_available()!=-1) numaNodes = numa_max_node()+1;
#endif

        std::vector<std::thread> workers;
        workers.reserve(nThreads);
        for (size_t i=0;i<nThreads;++i){
            int node = (numaNodes? int(i%numaNodes) : 0);
            workers.emplace_back(worker_entry, std::ref(ctx), int(i), node);
        }
        for (auto& t:workers) t.join();

        if (!ctx.errs.empty()) {
            std::string msg("save_bl_tif errors:\n");
            for (auto& e:ctx.errs) msg+="  - "+e+'\n';
            mexErrMsgIdAndTxt("save_bl_tif:Runtime", "%s", msg.c_str());
        }
        if (nlhs) plhs[0] = const_cast<mxArray*>(prhs[0]);
    }
    catch (const std::exception& e) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception","%s",e.what());
    }
}
