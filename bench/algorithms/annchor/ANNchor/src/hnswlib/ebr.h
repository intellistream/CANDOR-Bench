/*
 * Epoch-Based Reclamation (EBR)
 *
 * Based on libqsbr by Mindaugas Rasiukevicius (BSD-2-Clause License)
 * https://github.com/rmind/libqsbr
 *
 * Converted to header-only C++ for ANNchor project.
 *
 * Reference:
 *   K. Fraser, Practical lock-freedom,
 *   Technical Report UCAM-CL-TR-579, February 2004
 *   https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-579.pdf
 */

#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace annchor {

// Number of epochs (only 3 needed: e, e-1, e-2)
static constexpr unsigned EBR_EPOCHS = 3;

class EpochBasedReclamation {
   public:
    static constexpr uint32_t ACTIVE_FLAG = 0x80000000U;

   private:
    struct alignas(64) ThreadState {
        std::atomic<uint32_t> local_epoch{0};
    };

    std::atomic<uint32_t> global_epoch_{0};
    std::mutex list_lock_;
    std::vector<std::unique_ptr<ThreadState>> thread_states_;

    // Thread-local index into thread_states_
    static thread_local int tls_index_;
    static thread_local EpochBasedReclamation* tls_ebr_;

   public:
    EpochBasedReclamation() = default;
    ~EpochBasedReclamation() = default;

    // Non-copyable
    EpochBasedReclamation(const EpochBasedReclamation&) = delete;
    EpochBasedReclamation& operator=(const EpochBasedReclamation&) = delete;

    /**
     * Register the current thread for EBR.
     * Must be called once per thread before using enter/exit.
     */
    void register_thread() {
        std::lock_guard<std::mutex> lock(list_lock_);
        tls_index_ = static_cast<int>(thread_states_.size());
        tls_ebr_ = this;
        thread_states_.push_back(std::make_unique<ThreadState>());
    }

    /**
     * Unregister the current thread.
     */
    void unregister_thread() {
        if (tls_index_ >= 0 && tls_ebr_ == this) {
            thread_states_[tls_index_]->local_epoch.store(
                0, std::memory_order_relaxed);
            tls_index_ = -1;
            tls_ebr_ = nullptr;
        }
    }

    /**
     * Enter critical section - mark thread as active reader.
     */
    void enter() {
        assert(tls_index_ >= 0 && tls_ebr_ == this);
        ThreadState* ts = thread_states_[tls_index_].get();
        uint32_t epoch =
            global_epoch_.load(std::memory_order_relaxed) | ACTIVE_FLAG;
        ts->local_epoch.store(epoch, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    /**
     * Exit critical section - mark thread as inactive.
     */
    void exit() {
        assert(tls_index_ >= 0 && tls_ebr_ == this);
        ThreadState* ts = thread_states_[tls_index_].get();
        assert(ts->local_epoch.load(std::memory_order_relaxed) & ACTIVE_FLAG);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        ts->local_epoch.store(0, std::memory_order_relaxed);
    }

    /**
     * RAII guard for enter/exit.
     */
    class Guard {
        EpochBasedReclamation* ebr_;

       public:
        explicit Guard(EpochBasedReclamation* ebr) : ebr_(ebr) {
            ebr_->enter();
        }
        ~Guard() { ebr_->exit(); }
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
    };

    Guard guard() { return Guard(this); }

    /**
     * Try to synchronize and advance to a new epoch.
     * @param gc_epoch Output: epoch that is safe to reclaim
     * @return true if a new epoch was announced
     */
    bool sync(unsigned& gc_epoch) {
        uint32_t epoch = global_epoch_.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        {
            std::lock_guard<std::mutex> lock(list_lock_);
            for (auto& ts : thread_states_) {
                uint32_t local =
                    ts->local_epoch.load(std::memory_order_relaxed);
                bool active = (local & ACTIVE_FLAG) != 0;
                if (active && local != (epoch | ACTIVE_FLAG)) {
                    gc_epoch = gc_epoch_internal();
                    return false;
                }
            }
        }

        global_epoch_.store((epoch + 1) % EBR_EPOCHS,
                            std::memory_order_relaxed);
        gc_epoch = gc_epoch_internal();
        return true;
    }

    unsigned staging_epoch() const {
        return global_epoch_.load(std::memory_order_relaxed);
    }

    unsigned gc_epoch() const { return gc_epoch_internal(); }

    void full_sync() {
        unsigned target = staging_epoch();
        unsigned epoch;
        while (true) {
            if (sync(epoch) && epoch == target) return;
            std::this_thread::yield();
        }
    }

    bool in_critical() const {
        if (tls_index_ < 0 || tls_ebr_ != this) return false;
        return (thread_states_[tls_index_]->local_epoch.load(
                    std::memory_order_relaxed) &
                ACTIVE_FLAG) != 0;
    }

   private:
    unsigned gc_epoch_internal() const {
        return (global_epoch_.load(std::memory_order_relaxed) + 1) % EBR_EPOCHS;
    }
};

// Thread-local storage definitions
inline thread_local int EpochBasedReclamation::tls_index_ = -1;
inline thread_local EpochBasedReclamation* EpochBasedReclamation::tls_ebr_ =
    nullptr;

}  // namespace annchor
