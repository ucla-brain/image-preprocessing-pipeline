#include "mex_thread_utils.hpp"

#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

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
    // Fallback: pair remaining unused PUs within each NUMA node
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

// ===============
// get_cpu_busy_jiffies: (Linux only!)
// ===============
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
            // busy = user + nice + system + irq + softirq + steal
            uint64_t busy = user + nice + system + irq + softirq + steal;
            cpu_busy[puIdx] = busy;
        }
    }
    return cpu_busy;
}
#endif

// ===============
// find_least_busy_numa_node: Linux uses CPU jiffies, others pick node 0
// ===============
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

// --- MAIN FUNCTION: assign all pairs on one NUMA node ---
std::vector<ThreadAffinityPair>
assign_thread_affinity_pairs_single_numa(std::size_t maxPairs, unsigned numaNode)
{
    hwloc_topology_t topology = g_hwlocTopo->get();

    // Does the topology expose any NUMA objects at all?
    const int nbNumaObjs = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);
    const bool hasRealNuma = (nbNumaObjs > 0);

    // ---------------------------------------------------------------------
    // 1. Collect all PU indices that belong to the requested NUMA node
    //    • If there are no NUMA objects, treat the entire machine as node 0.
    // ---------------------------------------------------------------------
    std::vector<unsigned> pus;       // logical PU indices
    std::vector<unsigned> sockets;   // their package (socket) indices

    const int totalPU = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);

    for (int i = 0; i < totalPU; ++i) {
        hwloc_obj_t pu = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, i);
        if (!pu) continue;

        hwloc_obj_t node = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_NUMANODE, pu);

        unsigned nodeId = (node ? node->os_index : 0u);   // synthetic node 0
        if (!hasRealNuma) nodeId = 0;                     // whole machine = node 0

        if (nodeId == numaNode) {
            pus.push_back(pu->os_index);

            hwloc_obj_t sock = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_PACKAGE, pu);
            sockets.push_back(sock ? sock->os_index : 0u);
        }
    }

    // ---------------------------------------------------------------------
    // 2. Build PU pairs (SMT siblings in-order)
    // ---------------------------------------------------------------------
    std::size_t maxPossiblePairs = pus.size() / 2;
    if (maxPairs > maxPossiblePairs) {
#ifdef MEX_FILE
        mexWarnMsgIdAndTxt("assign_thread_affinity_pairs_single_numa:oversub",
            "Requested %zu pairs but NUMA node %u only has %zu pairs; "
            "returning the maximum available.",
            maxPairs, numaNode, maxPossiblePairs);
#endif
        maxPairs = maxPossiblePairs;
    }

    std::vector<ThreadAffinityPair> pairs;
    pairs.reserve(maxPairs);

    for (std::size_t i = 0; i + 1 < pus.size() && pairs.size() < maxPairs; i += 2) {
        pairs.push_back({pus[i], pus[i + 1], numaNode, sockets[i]});
    }

    // Fallback for extreme edge-case: only one PU on the node
    if (pairs.empty() && !pus.empty() && maxPairs > 0) {
        pairs.push_back({pus[0], pus[0], numaNode, sockets[0]});
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

// ==============================
//   allocate/free NUMA buffer
// ==============================
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
