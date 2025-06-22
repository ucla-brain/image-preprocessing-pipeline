/*==============================================================================
  save_bl_tif.cpp  (experimental‑opt patched)
  ------------------------------------------------------------------------------
  High‑throughput Z‑slice saver for 3‑D MATLAB arrays (one TIFF per slice).

  VERSION  : 2025‑06‑21‑exp‑r1  (thread‑local TIFF reuse, chunked fetch,
                                NUMA‑local hugepages, raw gather‑write path)
  AUTHOR   : Keivan Moradi  (with ChatGPT‑o assistance)
  LICENSE  : GNU GPL v3   <https://www.gnu.org/licenses/>
==============================================================================*/

#include "mex.h"
#include "tiffio.h"

#include <atomic>
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
#  include <unistd.h>
#  include <sys/mman.h>
#  include <sched.h>
#  include <numa.h>
#  ifdef __has_include
#    if __has_include(<numaif.h>)
#      include <numaif.h>   // for mbind / MPOL_BIND
#    endif
#  endif
#endif

#if defined(_WIN32)
#  include <windows.h>
#endif

// Uncomment to enable thread pinning for benchmarking
//#define PIN_THREADS

/* ───────────────────────── Utility: NUMA‑aware / hugepage alloc ─────────── */
namespace {

static void* alloc_on_node(size_t bytes, int node, bool wantHuge)
{
#if defined(__linux__)
    // Try hugepage mmap first (2 MiB aligned)
    if (wantHuge && bytes >= (2UL << 20)) {
        size_t hugeSz = ((bytes + (2UL << 20) - 1) >> 21) << 21; // round‑up to 2 MiB
        void* p = ::mmap(nullptr, hugeSz, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p != MAP_FAILED) {
#       if defined(MPOL_BIND)
            unsigned long nodemask = 1UL << node;
            ::mbind(p, hugeSz, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, 0);
#       endif
            return p;
        }
    }
    // Fallback: libnuma allocator on requested node
    void* p = ::numa_alloc_onnode(bytes, node);
    if (p) return p;
#endif
    // Generic aligned allocation fallback
    void* p = nullptr;
    constexpr size_t ALIGN = 64;
    if (::posix_memalign(&p, ALIGN, ((bytes + ALIGN - 1) / ALIGN) * ALIGN) != 0)
        p = nullptr;
    return p;
}

static void free_on_node(void* p, size_t bytes, bool huge)
{
#if defined(__linux__)
    if (huge && p) {
        ::munmap(p, bytes); return;
    }
    if (p) {
        ::numa_free(p, bytes); return;
    }
#else
    (void)huge; (void)bytes;
#endif
    std::free(p);
}

} // unnamed namespace

/* ───────────────────────────── TASK DESCRIPTION ─────────────────────────── */
struct SaveTask {
    const uint8_t* basePtr;       // Start of whole volume
    size_t         sliceOffset;   // Byte‑offset of this slice
    mwSize         rows, cols;    // MATLAB dims *after* any transpose
    std::string    filePath;      // Destination path
    bool           alreadyXYZ;    // True if input is [X Y Z]
    mxClassID      classId;
    uint16_t       compressionTag;
    size_t         bytesPerSlice;
    size_t         bytesPerPixel;
    size_t         sliceIndex;
};

/* ───────────── Thread‑local scratch & TIFF handle (reuse between slices) ─── */
static thread_local uint8_t* scratch_buf   = nullptr;
static thread_local size_t   scratch_bytes = 0;
static thread_local bool     scratch_huge  = false;

struct TiffLocal {
    TIFF* tif          = nullptr;
    bool  need_close   = false;      // reuse across slices only if compressed
    std::string tmpPath;

    void ensure_open(const std::string& pathTmp) {
        if (!tif) {
            tif = TIFFOpen(pathTmp.c_str(), "w");
            if (!tif) throw std::runtime_error("Cannot open tmp " + pathTmp);
            tmpPath   = pathTmp;
            need_close = true;
        }
    }

    ~TiffLocal() {
        if (tif && need_close) TIFFClose(tif);
    }
};
static thread_local std::unique_ptr<TiffLocal> tl_tiff;

static void ensure_scratch(size_t wantBytes, int node)
{
    if (scratch_bytes >= wantBytes && scratch_buf) return;
    if (scratch_buf) free_on_node(scratch_buf, scratch_bytes, scratch_huge);

    bool askHuge = true;
    scratch_buf   = static_cast<uint8_t*>(alloc_on_node(wantBytes, node, askHuge));
    scratch_huge  = (askHuge && scratch_buf != nullptr);
    scratch_bytes = wantBytes;
    if (!scratch_buf) throw std::bad_alloc();
}

/* ─────────────────────────── Raw gather‑write helpers ───────────────────── */
namespace rawtiff {
#pragma pack(push,1)
struct Header { uint16_t le; uint16_t magic; uint32_t ifdOff; };
#pragma pack(pop)
static constexpr uint8_t ifdBlob[] = {
    // minimal IFD with one strip, uncompressed, 8/16‑bit, minisblack
    0x0E,0x00,              // 14 tags
    // TAG 256 ImageWidth   LONG 1 *
    0x00,0x01, 0x00,0x04, 0x00,0x00,0x00,0x01, 0,0,0,0,
    // TAG 257 ImageLength  LONG 1 *
    0x01,0x01, 0x00,0x04, 0x00,0x00,0x00,0x01, 0,0,0,0,
    // TAG 258 BitsPerSample SHORT 1 *
    0x02,0x01, 0x00,0x03, 0x00,0x00,0x00,0x01, 0,0,       0,0,
    // TAG 259 Compression  SHORT 1 = 1 (none)
    0x03,0x01, 0x00,0x03, 0x00,0x00,0x00,0x01, 0x01,0x00, 0,0,
    // TAG 262 Photometric  SHORT 1 = 1 (minisblack)
    0x06,0x01, 0x00,0x03, 0x00,0x00,0x00,0x01, 0x01,0x00, 0,0,
    // TAG 273 StripOffsets LONG 1 *
    0x11,0x01, 0x00,0x04, 0x00,0x00,0x00,0x01, 0,0,0,0,
    // TAG 277 SamplesPerPixel SHORT 1 =1
    0x15,0x01, 0x00,0x03, 0x00,0x00,0x00,0x01, 0x01,0x00, 0,0,
    // TAG 278 RowsPerStrip  LONG 1 *
    0x16,0x01, 0x00,0x04, 0x00,0x00,0x00,0x01, 0,0,0,0,
    // TAG 279 StripByteCounts LONG 1 *
    0x17,0x01, 0x00,0x04, 0x00,0x00,0x00,0x01, 0,0,0,0,
    // TAG 282 XRes RATIONAL 1 (72)
    0x1A,0x01, 0x00,0x05, 0x00,0x00,0x00,0x01, 0,0,0,0,
    // TAG 283 YRes
    0x1B,0x01, 0x00,0x05, 0x00,0x00,0x00,0x01, 0,0,0,0,
    // TAG 296 ResUnit SHORT 1 =2 (inch)
    0x28,0x01, 0x00,0x03, 0x00,0x00,0x00,0x01, 0x02,0x00, 0,0,
    // TAG 305 Software ASCII 1 "M"
    0x31,0x01, 0x00,0x02, 0x00,0x00,0x00,0x01, 'M',0,0,0,
    // TAG 339 Offset to next IFD =0
    0x00,0x00,0x00,0x00
};
} // rawtiff

/* ───────────────────────────── LOW‑LEVEL TIFF WRITE ─────────────────────── */
static void save_slice(const SaveTask& t, int numaNode)
{
    const mwSize srcRows = t.alreadyXYZ ? t.cols : t.rows;
    const mwSize srcCols = t.alreadyXYZ ? t.rows : t.cols;

    const uint8_t* src = t.basePtr + t.sliceOffset;
    const bool directWrite = (t.compressionTag == COMPRESSION_NONE && t.alreadyXYZ);

    const std::string tmpPath = t.filePath + ".tmp";

    // Fast path: raw gather‑write
    if (directWrite) {
        // Build 8‑byte header + IFD clone into stack buffer
        rawtiff::Header hdr{0x4949, 42, sizeof(hdr)};   // little‑endian
        // Fix up width/height/bits/offsets… by mutating a copy of IFD blob
        uint8_t ifd[sizeof(rawtiff::ifdBlob)];
        std::memcpy(ifd, rawtiff::ifdBlob, sizeof(ifd));
        auto store32 = [](uint8_t* p, uint32_t v){ std::memcpy(p,&v,4);} ;
        auto store16 = [](uint8_t* p, uint16_t v){ std::memcpy(p,&v,2);} ;
        // width  (tag 256) @ +8  (little‑endian)
        store32(ifd+8+8, static_cast<uint32_t>(srcCols));
        // height (tag 257)
        store32(ifd+20+8, static_cast<uint32_t>(srcRows));
        // bits/sample (tag 258)
        store16(ifd+32+8, t.bytesPerPixel==2?16:8);
        // strip offset (tag 273)
        uint32_t stripOff = sizeof(hdr)+sizeof(ifd);
        store32(ifd+56+8, stripOff);
        // rows/strip (tag 278)
        store32(ifd+76+8, static_cast<uint32_t>(srcRows));
        // byteCounts (tag 279)
        store32(ifd+88+8, static_cast<uint32_t>(t.bytesPerSlice));
        // total file size = header+ifd+pixels

        int fd = ::open(tmpPath.c_str(), O_CREAT|O_WRONLY|O_TRUNC|O_CLOEXEC, 0666);
        if (fd < 0) throw std::runtime_error("open " + tmpPath);
        // ensure size (ignore ftruncate error)
        (void)::ftruncate(fd, 0);
        struct iovec iov[3]{{&hdr,sizeof(hdr)},{ifd,sizeof(ifd)},{const_cast<uint8_t*>(src),(size_t)t.bytesPerSlice}};
        ssize_t n = ::writev(fd, iov, 3);
        if (n < 0 || (size_t)n != sizeof(hdr)+sizeof(ifd)+t.bytesPerSlice) {
            ::close(fd); ::unlink(tmpPath.c_str());
            throw std::runtime_error("writev failed on " + tmpPath);
        }
        ::close(fd);
        if (::rename(tmpPath.c_str(), t.filePath.c_str()) != 0) {
            ::unlink(tmpPath.c_str());
            throw std::runtime_error("rename failed " + tmpPath);
        }
        return;
    }

    // ───────── Slow path: compressed or transpose needed ──────── //
    if (!tl_tiff) tl_tiff = std::make_unique<TiffLocal>();
    tl_tiff->ensure_open(tmpPath);
    TIFF* tif = tl_tiff->tif;

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,  srcCols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, srcRows);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, t.bytesPerPixel==2?16:8);
    TIFFSetField(tif, TIFFTAG_COMPRESSION,   t.compressionTag);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,   PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG,  PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,  srcRows);

    // Prepare IO buffer (transpose if needed)
    ensure_scratch(t.bytesPerSlice, numaNode);
    uint8_t* dst = scratch_buf;
    if (!t.alreadyXYZ) {
        for (mwSize col=0; col<srcCols; ++col) {
            const uint8_t* srcCol = src + col * t.rows * t.bytesPerPixel;
            for (mwSize row=0; row<srcRows; ++row) {
                size_t di = (static_cast<size_t>(row)*srcCols + col)*t.bytesPerPixel;
                std::memcpy(dst+di, srcCol+row*t.bytesPerPixel, t.bytesPerPixel);
            }
        }
    } else {
        std::memcpy(dst, src, t.bytesPerSlice);
    }

    tsize_t nWritten = (t.compressionTag==COMPRESSION_NONE)
        ? TIFFWriteRawStrip(tif,0,dst,t.bytesPerSlice)
        : TIFFWriteEncodedStrip(tif,0,dst,t.bytesPerSlice);
    if (nWritten < 0) {
        ::unlink(tmpPath.c_str());
        throw std::runtime_error("TIFF write failed on slice " + std::to_string(t.sliceIndex));
    }
    TIFFRewriteDirectory(tif);
    TIFFCheckpointDirectory(tif);

    if (::rename(tmpPath.c_str(), t.filePath.c_str()) != 0) {
        ::unlink(tmpPath.c_str());
        throw std::runtime_error("rename failed on " + tmpPath);
    }
}

/* ─────────────────────────── Dispatch context / worker ─────────────────── */
struct CallContext {
    std::shared_ptr<const std::vector<SaveTask>> tasks;
    std::atomic_size_t nextChunk{0};
    std::mutex errMu; std::vector<std::string> errs;
};

static void worker_entry(CallContext& ctx,int tid,int numaNodes)
{
    int node = (numaNodes>0) ? tid%numaNodes : 0;
#if defined(__linux__)
    if (numa_available()!=-1) numa_run_on_node(node);
#endif
    const auto& jobs=*ctx.tasks;
    const size_t CHUNK=8;
    for(;;){
        size_t base = ctx.nextChunk.fetch_add(CHUNK,std::memory_order_relaxed);
        if(base>=jobs.size()) break;
        size_t hi = std::min(base+CHUNK, jobs.size());
        for(size_t i=base;i<hi;++i){
            try{ save_slice(jobs[i], node);}catch(const std::exception& e){
                std::lock_guard<std::mutex>lk(ctx.errMu); ctx.errs.emplace_back(e.what()); }
        }
    }
}

/* ──────────────────────────────── MEX ENTRY ─────────────────────────────── */
void mexFunction(int nlhs,mxArray*plhs[],int nrhs,const mxArray*prhs[])
{
    try {
        if(nrhs!=4 && nrhs!=5) mexErrMsgIdAndTxt("save_bl_tif:Input","Usage: save_bl_tif(volume,fileList,orderFlag,compression,[nThreads])");
        const mxArray* V = prhs[0];
        if(!mxIsUint8(V)&&!mxIsUint16(V)) mexErrMsgIdAndTxt("save_bl_tif:Input","Volume must be uint8/uint16");
        const mwSize* dims = mxGetDimensions(V);
        size_t dim0=dims[0], dim1=dims[1];
        size_t dim2=(mxGetNumberOfDimensions(V)==3)?dims[2]:1;
        const uint8_t* base = static_cast<const uint8_t*>(mxGetData(V));
        mxClassID cid = mxGetClassID(V);
        size_t bpp = (cid==mxUINT16_CLASS)?2:1;
        size_t bps = dim0*dim1*bpp;

        bool alreadyXYZ = mxIsLogicalScalarTrue(prhs[2]) || (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2])!=0.0);
        char* cstr = mxArrayToUTF8String(prhs[3]); if(!cstr) mexErrMsgIdAndTxt("save_bl_tif:Input","Invalid compression string");
        std::string comp(cstr); mxFree(cstr);
        uint16_t tag = COMPRESSION_NONE;
        if(comp=="lzw") tag=COMPRESSION_LZW; else if(comp=="deflate"||comp=="zip") tag=COMPRESSION_DEFLATE; else if(comp!="none") mexErrMsgIdAndTxt("save_bl_tif:Input","Compression must be 'none','lzw','deflate'");

        if(!mxIsCell(prhs[1])||mxGetNumberOfElements(prhs[1])!=dim2) mexErrMsgIdAndTxt("save_bl_tif:Input","fileList length mismatch");
        std::vector<std::string> paths(dim2);
        for(size_t i=0;i<dim2;++i){ mxArray* e=mxGetCell(prhs[1],i); if(!mxIsChar(e)) mexErrMsgIdAndTxt("save_bl_tif:Input","fileList items must be char"); char* s=mxArrayToUTF8String(e); paths[i]=s; mxFree(s);}

        auto tasks = std::make_shared<std::vector<SaveTask>>(); tasks->reserve(dim2);
        for(size_t i=0;i<dim2;++i) tasks->push_back({base,i*bps,dim0,dim1,paths[i],alreadyXYZ,cid,tag,bps,bpp,i});

        CallContext ctx; ctx.tasks=tasks;
        size_t hw=std::thread::hardware_concurrency(); if(hw==0) hw=tasks->size();
        size_t nThreads=std::min(hw,tasks->size());
        if(nrhs==5){ double req=mxGetScalar(prhs[4]); if(!(req>0)) mexErrMsgIdAndTxt("save_bl_tif:Input","nThreads positive"); nThreads=std::min<size_t>(req,tasks->size()); }

        size_t numaNodes=0;#if defined(__linux__) if(numa_available()!=-1) numaNodes=numa_max_node()+1;#endif
        std::vector<std::thread> workers; workers.reserve(nThreads);
        for(size_t t=0;t<nThreads;++t) workers.emplace_back([&,t]{ worker_entry(ctx,t,numaNodes);} );
        for(auto& th:workers) th.join();
        if(!ctx.errs.empty()){ std::string msg("save_bl_tif errors:\n"); for(auto&e:ctx.errs) msg+="  - "+e+'\n'; mexErrMsgIdAndTxt("save_bl_tif:Runtime","%s",msg.c_str()); }
        if(nlhs) plhs[0]=const_cast<mxArray*>(prhs[0]);
    } catch(const std::exception& e){ mexErrMsgIdAndTxt("save_bl_tif:Exception","%s",e.what()); }
}
