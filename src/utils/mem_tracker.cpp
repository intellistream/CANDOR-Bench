//
// Created by tony on 27/12/23.
//

#include <utils/mem_tracker.h>

namespace INTELLI {
INTELLI::MemoryTracker *gIns = nullptr;
void MemoryTracker::setActiveInstance(INTELLI::MemoryTracker *ins) {
  gIns = ins;
}
void MemoryTracker::sigHandler(int signo) {
  if (signo == SIGALRM) {
    // Trigger a memory sample when the timer expires
    // Since sigHandler is static, we can only access static members
    gIns->triggerMemorySample();
  }
}
} // namespace INTELLI