#ifndef CANDY_INCLUDE_CANDY_LEGACYCONCURRENTINDEX_H_
#define CANDY_INCLUDE_CANDY_LEGACYCONCURRENTINDEX_H_

#include <Utils/ConfigMap.hpp>
#include <memory>
#include <vector>
#include <tuple>
#include <Utils/IntelliTensorOP.hpp>
#include <CANDY/AbstractIndex.h>
#include <CANDY/IndexTable.h>

using BatchIndex = size_t;
using QueryIndex = size_t;

using SearchResults = std::vector<torch::Tensor>;
using SearchRecord = std::tuple<BatchIndex, QueryIndex, SearchResults>;

namespace CANDY {

class LegacyConcurrentIndex : public CANDY::AbstractIndex {
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

typedef std::shared_ptr<class CANDY::LegacyConcurrentIndex> LegacyConcurrentIndexPtr;

#define newLegacyConcurrentIndex std::make_shared<CANDY::LegacyConcurrentIndex>
}

#endif // CANDY_INCLUDE_CANDY_LEGACYCONCURRENTINDEX_H_
