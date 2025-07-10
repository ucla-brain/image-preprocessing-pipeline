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