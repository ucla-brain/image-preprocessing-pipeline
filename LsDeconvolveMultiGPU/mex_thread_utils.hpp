#ifndef MEX_THREAD_UTILS_HPP
#define MEX_THREAD_UTILS_HPP

#include <queue>
#include <mutex>
#include <condition_variable>
#include <set>
#include <map>
#include <memory>
#include <thread>
#include <cassert>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cstddef>

extern "C" {
#include <hwloc.h>
}

// =========================
//   BoundedQueue Template
// =========================
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t maxSize) : maximumSize_(maxSize) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(queueMutex_);
        queueNotFull_.wait(lock, [this] { return queue_.size() < maximumSize_; });
        queue_.emplace(std::move(item));
        queueNotEmpty_.notify_one();
    }

    void wait_and_pop(T& item) {
        std::unique_lock<std::mutex> lock(queueMutex_);
        queueNotEmpty_.wait(lock, [this] { return !queue_.empty(); });
        item = std::move(queue_.front());
        queue_.pop();
        queueNotFull_.notify_one();
    }

private:
    mutable std::mutex queueMutex_;
    std::condition_variable queueNotFull_;
    std::condition_variable queueNotEmpty_;
    std::queue<T> queue_;
    size_t maximumSize_;
};

// ===============================
//   HWLOC NUMA/Topology Helpers
// ===============================
class HwlocTopologyRAII {
public:
    HwlocTopologyRAII();
    ~HwlocTopologyRAII();
    hwloc_topology_t get() const noexcept;
private:
    hwloc_topology_t topology_;
};

// Global topology pointer (define in one .cpp)
extern std::unique_ptr<HwlocTopologyRAII> g_hwlocTopo;

// Platform-safe logical core count
size_t get_available_cores();

// Thread affinity helpers
struct ThreadAffinityPair {
    unsigned producerLogicalCore;
    unsigned consumerLogicalCore;
    unsigned numaNode;
    unsigned socket;
};

std::vector<ThreadAffinityPair> assign_thread_affinity_pairs(size_t pairCount);

// Choose NUMA node (returns os_index); on Windows picks first available node
unsigned find_least_busy_numa_node(hwloc_topology_t topology);

// Assign all pairs within a single NUMA node
std::vector<ThreadAffinityPair> assign_thread_affinity_pairs_single_numa(size_t maxPairs, unsigned numaNode);

// Set thread affinity using hwloc
void set_thread_affinity(unsigned logicalCoreId);

// One-time safe topology initialization
void ensure_hwloc_initialized();

// Allocate/frees NUMA-local memory (hwloc, cross-platform)
void* allocate_numa_local_buffer(hwloc_topology_t topology, size_t bytes);
void  free_numa_local_buffer(hwloc_topology_t topology, void* buf, size_t bytes);

// Returns first logical core (PU) on a given NUMA node, or -1 if none found
int get_first_core_on_numa_node(hwloc_topology_t topology, unsigned numaNode);

#endif // MEX_THREAD_UTILS_HPP
