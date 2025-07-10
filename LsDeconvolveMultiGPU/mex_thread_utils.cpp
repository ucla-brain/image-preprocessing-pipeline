#include "mex_thread_utils.hpp"

#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <vector>
#include <thread>
#include <memory>
#include <stdexcept>

// -----------------------------------------------
//        Topology Singleton
// -----------------------------------------------
std::unique_ptr<HwlocTopologyRAII> g_hwlocTopo;
std::once_flag g_hwlocTopoOnceFlag;

void ensure_hwloc_initialized() {
    std::call_once(g_hwlocTopoOnceFlag, []{
        g_hwlocTopo = std::make_unique<HwlocTopologyRAII>();
    });
}

// -----------------------------------------------
//        HwlocTopologyRAII Methods
// -----------------------------------------------
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

// ===============================================
//        NUMA/core helper implementations
// ===============================================
size_t get_available_cores() {
    auto hint = std::thread::hardware_concurrency();
    return hint ? static_cast<size_t>(hint) : 1;
}

// Return vector of all logical PUs (core IDs) on the least-busy NUMA node
std::vector<unsigned> get_cores_on_numa_node() {
    ensure_hwloc_initialized();
    hwloc_topology_t topology = g_hwlocTopo->get();
    unsigned numaNode = find_least_busy_numa_node(topology);

    const int nbNumaObjs = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);
    const bool hasRealNuma = (nbNumaObjs > 0);

    std::vector<unsigned> pus;
    const int totalPU = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
    for (int i = 0; i < totalPU; ++i) {
        hwloc_obj_t pu = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, i);
        if (!pu) continue;
        hwloc_obj_t node = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_NUMANODE, pu);
        unsigned nodeId = (node ? node->os_index : 0u);
        if (!hasRealNuma) nodeId = 0;
        if (nodeId == numaNode) {
            pus.push_back(pu->os_index);
        }
    }
    return pus;
}

// Return vector of SMT sibling pairs (producer/consumer) on least busy NUMA node
std::vector<ThreadAffinityPair>
assign_thread_affinity_pairs_single_numa(std::size_t maxPairs)
{
    ensure_hwloc_initialized();
    hwloc_topology_t topology = g_hwlocTopo->get();
    unsigned numaNode = find_least_busy_numa_node(topology);

    const int nbNumaObjs = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);
    const bool hasRealNuma = (nbNumaObjs > 0);

    std::vector<unsigned> pus, sockets;
    const int totalPU = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);

    for (int i = 0; i < totalPU; ++i) {
        hwloc_obj_t pu = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, i);
        if (!pu) continue;
        hwloc_obj_t node = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_NUMANODE, pu);

        unsigned nodeId = (node ? node->os_index : 0u);
        if (!hasRealNuma) nodeId = 0;

        if (nodeId == numaNode) {
            pus.push_back(pu->os_index);

            hwloc_obj_t sock = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_PACKAGE, pu);
            sockets.push_back(sock ? sock->os_index : 0u);
        }
    }

    std::size_t maxPossiblePairs = pus.size() / 2;
    if (maxPairs > maxPossiblePairs)
        maxPairs = maxPossiblePairs;

    std::vector<ThreadAffinityPair> pairs;
    pairs.reserve(maxPairs);

    for (std::size_t i = 0; i + 1 < pus.size() && pairs.size() < maxPairs; i += 2) {
        pairs.push_back({pus[i], pus[i + 1], numaNode, sockets[i]});
    }

    // Edge: only one PU
    if (pairs.empty() && !pus.empty() && maxPairs > 0) {
        pairs.push_back({pus[0], pus[0], numaNode, sockets[0]});
    }

    return pairs;
}

// ===============================================
//         Set thread affinity using hwloc
// ===============================================
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

// ===============================================
//         Find least busy NUMA node
// ===============================================
#if defined(__linux__)
static std::map<unsigned, uint64_t> get_cpu_busy_jiffies()
{
    std::ifstream stat("/proc/stat");
    std::string line;
    std::map<unsigned, uint64_t> cpu_busy;
    while (std::getline(stat, line)) {
        if (line.rfind("cpu", 0) == 0 && line.size() > 3 && std::isdigit(line[3])) {
            std::istringstream iss(line);
            std::string cpuLabel;
            iss >> cpuLabel;
            unsigned puIdx = std::stoi(cpuLabel.substr(3));
            uint64_t user, nice, system, idle, iowait, irq, softirq, steal = 0;
            iss >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
            uint64_t busy = user + nice + system + irq + softirq + steal;
            cpu_busy[puIdx] = busy;
        }
    }
    return cpu_busy;
}
#endif

unsigned find_least_busy_numa_node(hwloc_topology_t topology)
{
#if defined(__linux__)
    auto cpu_busy = get_cpu_busy_jiffies();

    std::map<unsigned, std::vector<unsigned>> nodeToPUs;
    int totalPU = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);

    for (int i = 0; i < totalPU; ++i) {
        hwloc_obj_t pu = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, i);
        if (!pu) continue;
        unsigned nodeId = 0;
        hwloc_obj_t node = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_NUMANODE, pu);
        if (node) nodeId = node->os_index;
        nodeToPUs[nodeId].push_back(pu->os_index);
    }

    unsigned bestNode = 0;
    uint64_t bestBusy = ~0ULL;
    for (const auto& [node, pus] : nodeToPUs) {
        uint64_t busySum = 0;
        for (unsigned pu : pus) {
            busySum += cpu_busy.count(pu) ? cpu_busy[pu] : 0;
        }
        if (busySum < bestBusy) {
            bestBusy = busySum;
            bestNode = node;
        }
    }
    return bestNode;
#else
    // On Windows/Mac: just pick node 0
    int n = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);
    if (n > 0) {
        hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, 0);
        return obj ? obj->os_index : 0;
    }
    return 0;
#endif
}

// ===============================================
//      Allocate/free NUMA-local buffer
// ===============================================
void* allocate_numa_local_buffer(hwloc_topology_t topology, size_t bytes, unsigned numaNode)
{
    hwloc_obj_t node = nullptr;
    int n = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);
    for (int i = 0; i < n; ++i) {
        hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, i);
        if (obj && obj->os_index == numaNode) {
            node = obj;
            break;
        }
    }
    if (!node) return nullptr;
    hwloc_nodeset_t nodeset = hwloc_bitmap_dup(node->nodeset);
    void* buf = hwloc_alloc_membind(
        topology, bytes, nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD);
    hwloc_bitmap_free(nodeset);
    return buf;
}

void free_numa_local_buffer(hwloc_topology_t topology, void* buf, size_t bytes)
{
    hwloc_free(topology, buf, bytes);
}

int get_first_core_on_numa_node(hwloc_topology_t topology, unsigned numaNode) {
    int totalPU = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
    for (int i = 0; i < totalPU; ++i) {
        hwloc_obj_t pu = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, i);
        hwloc_obj_t node = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_NUMANODE, pu);
        if (node && node->os_index == numaNode)
            return static_cast<int>(pu->os_index);
    }
    return -1;
}
