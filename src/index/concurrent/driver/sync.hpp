#pragma once

// Concurrency primitives for the benchmark driver: a token-bucket rate
// limiter and a bounded MPMC queue with channel semantics — capacity 0 is
// an unbuffered rendezvous where push blocks until a consumer receives.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace candor {
namespace driver {

using Clock = std::chrono::steady_clock;

class TokenBucket {
   public:
    TokenBucket(double rate, double burst)
        : rate_(rate), burst_(burst), tokens_(burst), last_(Clock::now()) {}

    void wait() {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            refill();
            if (tokens_ >= 1.0) {
                tokens_ -= 1.0;
                return;
            }
            double need_sec = (1.0 - tokens_) / rate_;
            lk.unlock();
            std::this_thread::sleep_for(
                std::chrono::duration<double>(need_sec));
            lk.lock();
        }
    }

   private:
    void refill() {
        auto now = Clock::now();
        double dt = std::chrono::duration<double>(now - last_).count();
        last_ = now;
        tokens_ = std::min(burst_, tokens_ + dt * rate_);
    }

    double rate_;
    double burst_;
    double tokens_;
    Clock::time_point last_;
    std::mutex mu_;
};

// rate = points/sec across batches of batch_size; rate <= 0 means no
// limiter. Burst is batch_size waits, starting full.
inline std::unique_ptr<TokenBucket> build_limiter(double event_rate,
                                                  size_t batch_size) {
    if (event_rate <= 0) return nullptr;
    if (batch_size == 0) batch_size = 1;
    double per_batch = event_rate / static_cast<double>(batch_size);
    if (per_batch <= 0) return nullptr;
    return std::make_unique<TokenBucket>(per_batch,
                                         static_cast<double>(batch_size));
}

template <typename T>
class BoundedQueue {
   public:
    explicit BoundedQueue(size_t cap) : cap_(cap) {}

    void push(T&& v) {
        std::unique_lock<std::mutex> lk(mu_);
        if (cap_ == 0) {
            not_full_.wait(lk, [&] { return q_.empty() || closed_; });
            if (closed_) return;
            uint64_t my_seq = ++push_seq_;
            q_.push_back(std::move(v));
            not_empty_.notify_one();
            taken_.wait(lk, [&] { return pop_seq_ >= my_seq || closed_; });
            return;
        }
        not_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
        if (closed_) return;
        q_.push_back(std::move(v));
        not_empty_.notify_one();
    }

    // Returns false once the queue is closed and drained.
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(mu_);
        not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        if (cap_ == 0) {
            ++pop_seq_;
            taken_.notify_all();
        }
        not_full_.notify_one();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
        taken_.notify_all();
    }

   private:
    std::deque<T> q_;
    size_t cap_;
    bool closed_ = false;
    uint64_t push_seq_ = 0, pop_seq_ = 0;
    std::mutex mu_;
    std::condition_variable not_empty_, not_full_, taken_;
};

}  // namespace driver
}  // namespace candor
