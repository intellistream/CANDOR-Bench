//
// Created by tony on 10/05/23.
//

#include <data_loader/abstract_data_loader.h>
#include <utils/intelli_log.h>
//do nothing in abstract class
using namespace std;

bool CANDOR::AbstractDataLoader::hijackConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  return true;
}
bool CANDOR::AbstractDataLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);

  return true;
}

torch::Tensor CANDOR::AbstractDataLoader::getData() {
  return torch::rand({1, 1});
}

torch::Tensor CANDOR::AbstractDataLoader::getQuery() {
  return torch::rand({1, 1});
}

torch::Tensor CANDOR::AbstractDataLoader::getDataAt(int64_t startPos, int64_t endPos) {
  auto ru = getData();
  return ru.slice(0, startPos, endPos).nan_to_num(0);
}
torch::Tensor CANDOR::AbstractDataLoader::getQueryAt(int64_t startPos, int64_t endPos) {
  auto ru = getQuery();
  return ru.slice(0, startPos, endPos).nan_to_num(0);
}
int64_t CANDOR::AbstractDataLoader::size() {
  auto ru = getData();
  return ru.size(0);
}
int64_t CANDOR::AbstractDataLoader::getDimension() {
  auto ru = getData();
  return ru.size(1);
}