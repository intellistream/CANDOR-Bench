#pragma once

#include <string.h>
#include <cstdint>
#include <deque>
#include <mutex>
#include <ostream>
#include <vector>

namespace hnswlib {
typedef unsigned short int vl_type;

class VisitedList {
   public:
    vl_type curV;
    vl_type* mass;
    unsigned int numelements;

    VisitedList(int numelements1) {
        curV = -1;
        numelements = numelements1;
        mass = new vl_type[numelements];
    }

    void reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    }

    ~VisitedList() {
        delete[] mass;
    }
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    std::deque<VisitedList*> pool;
    std::recursive_mutex poolguard;
    int numelements;
    std::vector<VisitedList*> all_lists_;

   public:
    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++) {
            VisitedList* vl = new VisitedList(numelements);
            all_lists_.push_back(vl);
            pool.push_front(vl);
        }
    }

    VisitedList* getFreeVisitedList() {
        VisitedList* rez;
        {
            std::unique_lock<std::recursive_mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                rez = new VisitedList(numelements);
                all_lists_.push_back(rez);
            }
        }
        rez->reset();
        return rez;
    }

    void releaseVisitedList(VisitedList* vl) {
        std::unique_lock<std::recursive_mutex> lock(poolguard);
        pool.push_front(vl);
    }

    size_t getMemoryUsage() const {
        size_t total_memory = 0;

        total_memory += sizeof(VisitedListPool);
        total_memory += all_lists_.size() * sizeof(VisitedList*);

        for (const auto& vl : all_lists_) {
            if (vl) {
                total_memory += sizeof(VisitedList);
                total_memory += vl->numelements * sizeof(vl_type);
            }
        }

        return total_memory;
    }

    void dumpAddressRanges(std::ostream& out, const char* role) const {
        auto write_range = [&](const char* cls, size_t id, uintptr_t begin,
                               uintptr_t end) {
            if (begin == 0 || end <= begin) return;
            out << cls << '_' << role << ',' << id << ",0,0x" << std::hex
                << begin << ",0x" << end << std::dec << ',' << (end - begin)
                << '\n';
        };

        const uintptr_t pool_begin = reinterpret_cast<uintptr_t>(this);
        write_range("visited_pool_object", 0, pool_begin,
                    pool_begin + sizeof(*this));
        const uintptr_t deque_begin = reinterpret_cast<uintptr_t>(&pool);
        write_range("visited_pool_deque", 0, deque_begin,
                    deque_begin + sizeof(pool));
        const uintptr_t guard_begin = reinterpret_cast<uintptr_t>(&poolguard);
        write_range("visited_pool_guard", 0, guard_begin,
                    guard_begin + sizeof(poolguard));

        for (size_t i = 0; i < all_lists_.size(); ++i) {
            const VisitedList* vl = all_lists_[i];
            if (vl == nullptr) continue;
            const uintptr_t list_begin = reinterpret_cast<uintptr_t>(vl);
            write_range("visited_list_object", i, list_begin,
                        list_begin + sizeof(*vl));
            const uintptr_t mass_begin = reinterpret_cast<uintptr_t>(vl->mass);
            write_range("visited_mass", i, mass_begin,
                        mass_begin +
                            static_cast<uintptr_t>(vl->numelements) *
                                sizeof(vl_type));
        }
    }

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList* rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }
};
}  // namespace hnswlib
