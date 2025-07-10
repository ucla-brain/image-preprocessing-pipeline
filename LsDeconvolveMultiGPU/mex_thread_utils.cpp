#include "mex_thread_utils.hpp"

#include <cstdlib>
#include <algorithm>
#include <iostream>

#if defined(_WIN32)
#include <windows.h>
#endif
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif

std::unique_ptr<HwlocTopologyRAII> g_hwlocTopo = nullptr;

// ===============================
//   HwlocTopologyRAII Definition
// ===============================
HwlocTopologyRAII::HwlocTopologyRAII() {
    if (hwloc_topology_init(&topology_) != 0)
        throw std::runtime_error("hwloc_topology_init failed");
    if (hwloc_topology_load(topology_) != 0)
        throw std::runtime_error("hwloc_topology_load failed");
}
HwlocTopologyRAII::~HwlocTopologyRAII() {
    if (topology_) hwloc_topology_destroy(topology_);
}
hwloc_topology_t HwlocTopologyRAII::get() const noexcept { return topology_; }

// ==============================
//   ensure_hwloc_initialized
// ==============================
void ensure_hwloc_initialized() {
    if (!g_hwlocTopo) {
        g_hwlocTopo = std::make_unique<HwlocTopologyRAII>();
    }
}

// ==============================
//   get_available_cores
// ==============================
size_t get_available_cores() {
#if defined(_WIN32)
    DWORD_PTR processMask = 0, systemMask = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask))
        return static_cast<size_t>(std::bitset<sizeof(processMask)*8>(processMask).count());
#elif defined(__linux__) || defined(__APPLE__)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n > 0) return static_cast<size_t>(n);
#endif
    auto hint = std::thread::hardware_concurrency();
    return hint ? static_cast<size_t>(hint) : 1;
}

// ==============================
//   assign_thread_affinity_pairs
// ==============================
std::vector<ThreadAffinityPair> assign_thread_affinity_pairs(size_t pairCount)
{
    ensure_hwloc_initialized();
    std::vector<ThreadAffinityPair> pairs;
    std::set<unsigned> usedPUs;
    hwloc_topology_t topology = g_hwlocTopo->get();

    int totalCores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);

    // SMT siblings first
    for (int i = 0; i < totalCores && pairs.size() < pairCount; ++i) {
        hwloc_obj_t core = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, i);
        unsigned nodeId = 0, socketId = 0;
        hwloc_obj_t node = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_NUMANODE, core);
        if (node) nodeId = node->os_index;
        hwloc_obj_t sock = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_PACKAGE, core);
        if (sock) socketId = sock->os_index;

        std::vector<unsigned> pus;
        for (unsigned j = 0; j < core->arity; ++j) {
            hwloc_obj_t child = core->children[j];
            if (child->type == HWLOC_OBJ_PU)
                pus.push_back(child->os_index);
        }
        if (pus.empty()) {
            hwloc_bitmap_t cpuset = hwloc_bitmap_dup(core->cpuset);
            hwloc_bitmap_singlify(cpuset);
            int puIdx = hwloc_bitmap_first(cpuset);
            if (puIdx >= 0)
                pus.push_back(static_cast<unsigned>(puIdx));
            hwloc_bitmap_free(cpuset);
        }
        if (pus.size() >= 2 && !usedPUs.count(pus[0]) && !usedPUs.count(pus[1])) {
            pairs.push_back({pus[0], pus[1], nodeId, socketId});
            usedPUs.insert(pus[0]);
            usedPUs.insert(pus[1]);
        } else if (pus.size() == 1 && !usedPUs.count(pus[0])) {
            pairs.push_back({pus[0], pus[0], nodeId, socketId});
            usedPUs.insert(pus[0]);
        }
    }
    // Second, pair remaining unused PUs within each NUMA node
    if (pairs.size() < pairCount) {
        int totalPU = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
        std::map<unsigned, std::vector<unsigned>> numaToPU;
        std::map<unsigned, std::vector<unsigned>> numaToSocket;
        for (int i = 0; i < totalPU; ++i) {
            hwloc_obj_t pu = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, i);
            if (usedPUs.count(pu->os_index)) continue;
            unsigned nodeId = 0, socketId = 0;
            hwloc_obj_t node = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_NUMANODE, pu);
            if (node) nodeId = node->os_index;
            hwloc_obj_t sock = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_PACKAGE, pu);
            if (sock) socketId = sock->os_index;
            numaToPU[nodeId].push_back(pu->os_index);
            numaToSocket[nodeId].push_back(socketId);
        }
        for (auto& [numaNode, pus] : numaToPU) {
            auto& sockets = numaToSocket[numaNode];
            for (size_t i = 0; i + 1 < pus.size() && pairs.size() < pairCount; i += 2) {
                pairs.push_back({pus[i], pus[i + 1], numaNode, sockets[i]});
                usedPUs.insert(pus[i]);
                usedPUs.insert(pus[i + 1]);
            }
        }
        std::vector<std::pair<unsigned, unsigned>> leftovers;
        for (auto& [numaNode, pus] : numaToPU) {
            auto& sockets = numaToSocket[numaNode];
            if (pus.size() % 2)
                leftovers.emplace_back(pus.back(), sockets.back());
        }
        for (size_t i = 0; i + 1 < leftovers.size() && pairs.size() < pairCount; i += 2) {
            pairs.push_back({leftovers[i].first, leftovers[i + 1].first, 0, leftovers[i].second});
            usedPUs.insert(leftovers[i].first);
            usedPUs.insert(leftovers[i + 1].first);
        }
    }
    if (pairs.size() < pairCount) {
        int totalPU = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
        std::vector<unsigned> allUnused;
        for (int i = 0; i < totalPU; ++i) {
            hwloc_obj_t pu = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, i);
            if (!usedPUs.count(pu->os_index)) allUnused.push_back(pu->os_index);
        }
        for (size_t i = 0; i + 1 < allUnused.size() && pairs.size() < pairCount; i += 2) {
            pairs.push_back({allUnused[i], allUnused[i + 1], 0, 0});
        }
    }
    return pairs;
}

// ==============================
//   set_thread_affinity
// ==============================
void set_thread_affinity(unsigned logicalCoreId)
{
    ensure_hwloc_initialized();
    hwloc_topology_t topology = g_hwlocTopo->get();
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_bitmap_zero(cpuset);
    hwloc_bitmap_set(cpuset, logicalCoreId);
    hwloc_set_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD);
    hwloc_bitmap_free(cpuset);
}


// -----------------------------------------------------------------------------
//  parallel_decode_and_copy  –  single-stage, one worker per physical core
// -----------------------------------------------------------------------------
void parallel_decode_and_copy(const std::vector<LoadTask>& tasks,
                              void*                        outData,
                              size_t                       bytesPerPixel)
{
    const size_t numSlices = tasks.size();

    // -------------------------------------------------------------------------
    // 1) How many threads?  -> nb physical cores if hwloc is present,
    //                         else half the logical PUs (crude SMT-2 guess)
    // -------------------------------------------------------------------------
    const unsigned logicalPUs = get_available_cores();     // already in utils

    unsigned physCoreCount = std::max(1u, logicalPUs / 2u);   // pessimistic fall-back

#if defined(HWLOC_API_VERSION)
    {
        hwloc_topology_t topo = nullptr;
        if (   hwloc_topology_init(&topo)  == 0
            && hwloc_topology_load(topo)   == 0)
        {
            unsigned cores = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE);
            if (cores) physCoreCount = cores;               // trust hwloc if it worked
            hwloc_topology_destroy(topo);
        }
    }
#endif

    const unsigned nThreads = static_cast<unsigned>(
        std::min<size_t>(numSlices, physCoreCount));

    const unsigned smtStride = (logicalPUs >= physCoreCount * 2) ? 2u : 1u;

    // -------------------------------------------------------------------------
    // 2) Simple static thread pool – each worker decodes + copies one slice
    // -------------------------------------------------------------------------
    std::atomic<uint32_t> nextIdx{0};
    std::vector<std::string> errors;
    std::mutex              errMtx;
    std::vector<std::thread> workers;
    workers.reserve(nThreads);

    for (unsigned t = 0; t < nThreads; ++t)
    {
        const unsigned pu = t * smtStride;          // PU #0 of core t
        workers.emplace_back([&, pu]()
        {
            set_thread_affinity(pu);                // helper from utils

            std::vector<uint8_t> tempBuf;           // per-thread tile/strip buffer

            while (true)
            {
                const uint32_t idx = nextIdx.fetch_add(1, std::memory_order_relaxed);
                if (idx >= tasks.size()) break;

                const LoadTask& task = tasks[idx];
                try
                {
                    //------------------------------------------------------------------
                    // Decode ROI of slice idx
                    //------------------------------------------------------------------
                    TiffHandle tif(TIFFOpen(task.path.c_str(), "r"));
                    if (!tif)
                        throw std::runtime_error("Cannot open file " + task.path);

                    const size_t sliceBytes =
                        static_cast<size_t>(task.cropH) * task.cropW * bytesPerPixel;
                    std::vector<uint8_t> sliceBuf(sliceBytes);

                    readSubRegionToBuffer(task, tif.get(),
                                          static_cast<uint8_t>(bytesPerPixel),
                                          sliceBuf, tempBuf);

                    //------------------------------------------------------------------
                    // Copy / transpose into MATLAB output
                    //------------------------------------------------------------------
                    uint8_t*       dstBase = static_cast<uint8_t*>(outData) +
                                             task.zIndex * task.pixelsPerSlice *
                                             bytesPerPixel;
                    const uint8_t* srcBase = sliceBuf.data();

                    if (!task.transpose)
                    {
                        const bool colMajor = (task.cropW < task.cropH);
                        if (colMajor)      // contiguous writes (dst)
                        {
                            const size_t dstColStride =
                                static_cast<size_t>(task.roiH) * bytesPerPixel;

                            if (bytesPerPixel == 2)
                            {
                                for (uint32_t col = 0; col < task.cropW; ++col)
                                {
                                    const uint16_t* src = reinterpret_cast<const uint16_t*>(srcBase) + col;
                                    uint16_t*       dst = reinterpret_cast<uint16_t*>(dstBase) + col * task.roiH;
                                    for (uint32_t row = 0; row < task.cropH; ++row)
                                        dst[row] = src[row * task.cropW];
                                }
                            }
                            else
                            {
                                for (uint32_t col = 0; col < task.cropW; ++col)
                                {
                                    const uint8_t* src = srcBase + col;
                                    uint8_t*       dst = dstBase + col * dstColStride;
                                    for (uint32_t row = 0; row < task.cropH; ++row)
                                        dst[row] = src[row * task.cropW];
                                }
                            }
                        }
                        else               // contiguous reads (src)
                        {
                            const size_t srcRowStride =
                                static_cast<size_t>(task.cropW) * bytesPerPixel;

                            if (bytesPerPixel == 2)
                            {
                                uint16_t*       dst16 = reinterpret_cast<uint16_t*>(dstBase);
                                const uint16_t* src16 = reinterpret_cast<const uint16_t*>(srcBase);
                                for (uint32_t row = 0; row < task.cropH; ++row)
                                {
                                    const uint16_t* srcRow = src16 + row * task.cropW;
                                    for (uint32_t col = 0; col < task.cropW; ++col)
                                        dst16[row + col * task.roiH] = srcRow[col];
                                }
                            }
                            else
                            {
                                for (uint32_t row = 0; row < task.cropH; ++row)
                                {
                                    const uint8_t* srcRow = srcBase + row * srcRowStride;
                                    for (uint32_t col = 0; col < task.cropW; ++col)
                                        dstBase[row + col * task.roiH] = srcRow[col];
                                }
                            }
                        }
                    }
                    else  // transpose path
                    {
                        const size_t rowBytes = static_cast<size_t>(task.cropW) * bytesPerPixel;
                        for (uint32_t row = 0; row < task.cropH; ++row)
                            std::memcpy(dstBase + row * rowBytes,
                                        srcBase + row * rowBytes,
                                        rowBytes);
                    }
                }
                catch (const std::exception& ex)
                {
                    std::lock_guard<std::mutex> g(errMtx);
                    errors.emplace_back("Slice " + std::to_string(task.zIndex+1) +
                                        ": " + ex.what());
                }
            }
        });
    }

    for (auto& th : workers) th.join();

    if (!errors.empty())
    {
        std::ostringstream oss;
        oss << "Errors during load_bl_tif:\n";
        for (auto& e : errors) oss << "  - " << e << '\n';
        mexErrMsgIdAndTxt("load_bl_tif:Error", "%s", oss.str().c_str());
    }
}

