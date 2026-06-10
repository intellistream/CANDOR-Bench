/*! \file AbstractIndex.cpp*/
//
// Created by tony on 25/05/23.
//

#include <index/abstract_index.h>
#include <utils/utility_functions.h>
#include <time.h>
#include <chrono>
#include <assert.h>
static std::vector<std::string> u64ObjectToStringObject(std::vector<uint64_t> &u64s) {
  std::vector<std::string> ru(u64s.size());
  for (size_t i = 0; i < u64s.size(); i++) {
    uint64_t u64i = u64s[i];
    const char *char_ptr = reinterpret_cast<const char *>(&u64i);
    ru[i] = std::string(char_ptr, sizeof(uint64_t));
  }
  return ru;
}
static std::vector<uint64_t> stringObjectToU64Object(std::vector<std::string> &strs) {
  std::vector<uint64_t> ru(strs.size());
  for (size_t i = 0; i < strs.size(); i++) {
    uint64_t u64i = 0;
    std::memcpy(&u64i, strs[i].data(), sizeof(uint64_t));
    ru[i] = u64i;
  }
  return ru;
}
void CANDOR::AbstractIndex::reset() {

}
bool CANDOR::AbstractIndex::offlineBuild(torch::Tensor &t) {
  return false;
}
bool CANDOR::AbstractIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  std::string metricType = cfg->tryString("metricType", "IP", true);
  faissMetric = faiss::METRIC_L2;
  if (metricType == "dot" || metricType == "IP" || metricType == "cossim") {
    faissMetric = faiss::METRIC_INNER_PRODUCT;
  }

  return true;
}
bool CANDOR::AbstractIndex::setConfigClass(INTELLI::ConfigMap cfg) {
  INTELLI::ConfigMapPtr cfgPtr = newConfigMap();
  cfgPtr->loadFrom(cfg);
  return setConfig(cfgPtr);
}

bool CANDOR::AbstractIndex::setFrozenLevel(int64_t frozenLv) {
  assert(frozenLv >= 0);
  return false;
}
bool CANDOR::AbstractIndex::insertTensor(torch::Tensor &t) {
  assert(t.size(1));
  return false;
}

bool CANDOR::AbstractIndex::insertTensorWithIds(std::vector<faiss::idx_t> ids, torch::Tensor &t){
  return insertTensor(t);
}
bool CANDOR::AbstractIndex::insertStringObject(torch::Tensor &t, std::vector<std::string> &strs) {
  assert(t.size(1));
  assert (strs.size());
  return false;
}
bool CANDOR::AbstractIndex::insertU64Object(torch::Tensor &t, std::vector<uint64_t> &u64s) {
  auto strVec = u64ObjectToStringObject(u64s);
  return insertStringObject(t, strVec);
}
bool CANDOR::AbstractIndex::loadInitialU64Object(torch::Tensor &t, std::vector<uint64_t> &u64s) {
  auto strVec = u64ObjectToStringObject(u64s);
  return loadInitialStringObject(t, strVec);
}
bool CANDOR::AbstractIndex::loadInitialTensor(torch::Tensor &t) {
  return insertTensor(t);
}

bool CANDOR::AbstractIndex::loadInitialTensorWithIds(std::vector<faiss::idx_t> ids, torch::Tensor &t) {
  return loadInitialTensor(t);
}

std::vector<std::tuple<size_t, size_t, std::vector<torch::Tensor>>> 
CANDOR::AbstractIndex::ccInsertAndSearchTensor(torch::Tensor &t, torch::Tensor &qt, int64_t k) {
  std::vector<torch::Tensor> ru(1);
  ru[0] = torch::rand({1, 1});

  std::tuple<size_t, size_t, std::vector<torch::Tensor>> tp(1, 1, ru);

  return {tp};  
}

bool CANDOR::AbstractIndex::loadInitialStringObject(torch::Tensor &t, std::vector<std::string> &strs) {
  return insertStringObject(t, strs);
}
bool CANDOR::AbstractIndex::deleteTensor(torch::Tensor &t, int64_t k) {
  assert(t.size(1));
  assert(k > 0);
  return false;
}

bool CANDOR::AbstractIndex::deleteIndex(std::vector<faiss::idx_t>){
    return false;
}

bool CANDOR::AbstractIndex::deleteU64Object(torch::Tensor &t, int64_t k) {
  return deleteStringObject(t, k);
}

bool CANDOR::AbstractIndex::deleteStringObject(torch::Tensor &t, int64_t k) {
  assert(t.size(1));
  assert(k > 0);
  return false;
}
bool CANDOR::AbstractIndex::reviseTensor(torch::Tensor &t, torch::Tensor &w) {
  assert(t.size(1) == w.size(1));
  return false;
}
std::vector<faiss::idx_t> CANDOR::AbstractIndex::searchIndex(torch::Tensor q, int64_t k) {
  assert(k > 0);
  assert(q.size(1));
  std::vector<faiss::idx_t> ru(1);
  return ru;
}
std::vector<faiss::idx_t> CANDOR::AbstractIndex::searchIndexParam(torch::Tensor q, int64_t k, int64_t param){
    return searchIndex(q,k);
}

std::vector<std::vector<std::string>> CANDOR::AbstractIndex::searchStringObject(torch::Tensor &q, int64_t k) {
  assert(k > 0);
  assert(q.size(1));
  std::vector<std::vector<std::string>> ru(1);
  ru[0] = std::vector<std::string>(1);
  ru[0][0] = "";
  return ru;
}
std::vector<std::vector<uint64_t >> CANDOR::AbstractIndex::searchU64Object(torch::Tensor &q, int64_t k) {
  auto ruS = searchStringObject(q, k);
  std::vector<std::vector<uint64_t >> ruU = std::vector<std::vector<uint64_t >>(ruS.size());
  for (size_t i = 0; i < ruU.size(); i++) {
    ruU[i] = stringObjectToU64Object(ruS[i]);
  }
  return ruU;
}
std::vector<torch::Tensor> CANDOR::AbstractIndex::getTensorByIndex(std::vector<faiss::idx_t> &idx, int64_t k) {
  assert(k > 0);
  assert(idx.size());
  std::vector<torch::Tensor> ru(1);
  ru[0] = torch::rand({1, 1});
  return ru;
}
torch::Tensor CANDOR::AbstractIndex::rawData() {
  return torch::rand({1, 1});
}

std::vector<torch::Tensor> CANDOR::AbstractIndex::searchTensor(torch::Tensor &q, int64_t k) {
  assert(k > 0);
  assert(q.size(1));
  std::vector<torch::Tensor> ru(1);
  ru[0] = torch::rand({1, 1});
  return ru;
}
bool CANDOR::AbstractIndex::startHPC() {
  return false;
}

bool CANDOR::AbstractIndex::endHPC() {
  return false;
}
bool CANDOR::AbstractIndex::waitPendingOperations() {
  return true;
}
std::tuple<std::vector<torch::Tensor>,
           std::vector<std::vector<std::string>>> CANDOR::AbstractIndex::searchTensorAndStringObject(torch::Tensor &q,
                                                                                                    int64_t k) {
  auto ruT = searchTensor(q, k);
  auto ruS = searchStringObject(q, k);
  std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<std::string>>> ru(ruT, ruS);
  return ru;
}
bool CANDOR::AbstractIndex::loadInitialTensorAndQueryDistribution(torch::Tensor &t, torch::Tensor &query) {
  assert(query.size(0) > 0);
  return loadInitialTensor(t);
}
bool CANDOR::AbstractIndex::resetIndexStatistics() {
  return false;
}
INTELLI::ConfigMapPtr CANDOR::AbstractIndex::getIndexStatistics() {
  auto ru = newConfigMap();
  ru->edit("hasExtraStatistics", (int64_t) 0);
  return ru;
}