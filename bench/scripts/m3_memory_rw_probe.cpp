#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Record {
  uint32_t worker;
  double finish_ms;
  double batch_us;
};

struct Args {
  std::string out = "m3_memory_rw_probe_latency.csv";
  std::string start_ns_out;
  int measured_workers = 8;
  int competing_workers = 40;
  int duration_ms = 12000;
  int batch = 128;
  int array_mib = 512;
  bool write_store = false;
};

uint64_t now_ns() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

uint64_t xorshift64(uint64_t &x) {
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  return x;
}

bool in_write_period(double elapsed_ms) {
  return (elapsed_ms >= 2500.0 && elapsed_ms < 3500.0) ||
         (elapsed_ms >= 6000.0 && elapsed_ms < 7000.0) ||
         (elapsed_ms >= 9500.0 && elapsed_ms < 10500.0);
}

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    auto take = [&](const char *name) -> const char * {
      if (std::strcmp(argv[i], name) != 0 || i + 1 >= argc) {
        return nullptr;
      }
      return argv[++i];
    };
    if (const char *v = take("--out")) {
      args.out = v;
    } else if (const char *v = take("--start-ns-out")) {
      args.start_ns_out = v;
    } else if (const char *v = take("--measured-workers")) {
      args.measured_workers = std::atoi(v);
    } else if (const char *v = take("--competing-workers")) {
      args.competing_workers = std::atoi(v);
    } else if (const char *v = take("--duration-ms")) {
      args.duration_ms = std::atoi(v);
    } else if (const char *v = take("--batch")) {
      args.batch = std::atoi(v);
    } else if (const char *v = take("--array-mib")) {
      args.array_mib = std::atoi(v);
    } else if (const char *v = take("--write-op")) {
      args.write_store = std::strcmp(v, "store") == 0;
    }
  }
  return args;
}

} // namespace

int main(int argc, char **argv) {
  Args args = parse_args(argc, argv);
  if (args.measured_workers <= 0 || args.competing_workers < 0 || args.duration_ms <= 0 || args.batch <= 0) {
    std::cerr << "invalid arguments\n";
    return 2;
  }

  size_t bytes = static_cast<size_t>(args.array_mib) * 1024ull * 1024ull;
  size_t elems = 1;
  while ((elems * sizeof(std::atomic<uint64_t>)) < bytes) {
    elems <<= 1;
  }
  std::vector<std::atomic<uint64_t>> data(elems);
  for (size_t i = 0; i < elems; ++i) {
    data[i].store(i, std::memory_order_relaxed);
  }
  const size_t mask = elems - 1;

  std::atomic<bool> stop{false};
  std::atomic<uint64_t> sink{0};
  const uint64_t start_ns = now_ns();
  if (!args.start_ns_out.empty()) {
    std::ofstream start_out(args.start_ns_out);
    start_out << start_ns << "\n";
  }
  std::vector<std::vector<Record>> records(args.measured_workers);
  std::vector<std::thread> threads;
  threads.reserve(args.measured_workers + args.competing_workers);

  for (int w = 0; w < args.measured_workers; ++w) {
    threads.emplace_back([&, w] {
      uint64_t rng = 0x9e3779b97f4a7c15ull ^ (static_cast<uint64_t>(w) * 0xbf58476d1ce4e5b9ull);
      std::vector<Record> local;
      local.reserve(200000);
      uint64_t local_sink = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        const uint64_t begin = now_ns();
        for (int i = 0; i < args.batch; ++i) {
          size_t idx = xorshift64(rng) & mask;
          local_sink += data[idx].load(std::memory_order_relaxed);
        }
        const uint64_t end = now_ns();
        local.push_back(Record{
            static_cast<uint32_t>(w),
            static_cast<double>(end - start_ns) / 1000000.0,
            static_cast<double>(end - begin) / 1000.0,
        });
      }
      sink.fetch_add(local_sink, std::memory_order_relaxed);
      records[w].swap(local);
    });
  }

  for (int w = 0; w < args.competing_workers; ++w) {
    threads.emplace_back([&, w] {
      uint64_t rng = 0xd6e8feb86659fd93ull ^ (static_cast<uint64_t>(w + 1000) * 0x94d049bb133111ebull);
      uint64_t local_sink = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        double elapsed_ms = static_cast<double>(now_ns() - start_ns) / 1000000.0;
        if (in_write_period(elapsed_ms)) {
          for (int i = 0; i < args.batch; ++i) {
            size_t idx = xorshift64(rng) & mask;
            if (args.write_store) {
              data[idx].store(rng + static_cast<uint64_t>(idx), std::memory_order_relaxed);
              local_sink += idx;
            } else {
              local_sink += data[idx].fetch_add(1, std::memory_order_relaxed);
            }
          }
        } else {
          for (int i = 0; i < args.batch; ++i) {
            size_t idx = xorshift64(rng) & mask;
            local_sink += data[idx].load(std::memory_order_relaxed);
          }
        }
      }
      sink.fetch_add(local_sink, std::memory_order_relaxed);
    });
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(args.duration_ms));
  stop.store(true, std::memory_order_relaxed);
  for (auto &thread : threads) {
    thread.join();
  }

  std::ofstream out(args.out);
  out << "worker,finish_ms,batch_us\n";
  for (const auto &worker_records : records) {
    for (const auto &record : worker_records) {
      out << record.worker << ',' << record.finish_ms << ',' << record.batch_us << '\n';
    }
  }
  out.close();

  std::ofstream periods(args.out + ".periods.csv");
  periods << "start_ms,end_ms,competing_mode\n";
  periods << "0,2500,read\n";
  periods << "2500,3500,write\n";
  periods << "3500,6000,read\n";
  periods << "6000,7000,write\n";
  periods << "7000,9500,read\n";
  periods << "9500,10500,write\n";
  periods << "10500," << args.duration_ms << ",read\n";
  periods.close();

  std::cerr << "sink=" << sink.load(std::memory_order_relaxed) << " elems=" << elems << "\n";
  return 0;
}
