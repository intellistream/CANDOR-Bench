#ifndef CANDOR_INCLUDE_CANDOR_LEGACYCONCURRENTINDEX_H_
#define CANDOR_INCLUDE_CANDOR_LEGACYCONCURRENTINDEX_H_

#include <utils/config_map.hpp>
#include <memory>
#include <vector>
#include <tuple>
#include <utils/intelli_tensor_op.hpp>
#include <index/abstract_index.h>
#include <index/index_table.h>

using BatchIndex = size_t;
using QueryIndex = size_t;

using SearchResults = std::vector<torch::Tensor>;
using SearchRecord = std::tuple<BatchIndex, QueryIndex, SearchResults>;

namespace CANDOR {

class LegacyConcurrentIndex : public CANDOR::AbstractIndex {
 protected:
  AbstractIndexPtr myIndexAlgo = nullptr;
  std::string myConfigString = "";

  int64_t vecDim = 0;
  double writeRatio = 0.0;
  int64_t numThreads = 1;
  int64_t batchSize = 0;

 public:
  LegacyConcurrentIndex() {

  }

  ~LegacyConcurrentIndex() {

  }

  virtual bool loadInitialTensor(torch::Tensor &t);

  virtual void reset();

  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

  virtual std::vector<SearchRecord> ccInsertAndSearchTensor(torch::Tensor &t, torch::Tensor &qt, int64_t k);

  virtual std::vector<torch::Tensor> searchTensor(torch::Tensor &q, int64_t k);
};

typedef std::shared_ptr<class CANDOR::LegacyConcurrentIndex> LegacyConcurrentIndexPtr;

#define newLegacyConcurrentIndex std::make_shared<CANDOR::LegacyConcurrentIndex>
}

#endif // CANDOR_INCLUDE_CANDOR_LEGACYCONCURRENTINDEX_H_
