//
// Created by Isshin on 2024/1/18.
//
#define CATCH_CONFIG_MAIN
#include <index/hnsw_naive/hnsw.h>
#include <index/hnsw_naive_index.h>
#include "catch.hpp"
#include <index.h>

using namespace std;
TEST_CASE("Test HNSWiNDEX", "[short]") {
  CANDOR::HNSWNaiveIndex hnswIdx;
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  cfg->edit("vecDim", (int64_t) 3);
  cfg->edit("M", (int64_t) 4);
  CANDOR::VisitedTable vt;
  hnswIdx.setConfig(cfg);
  auto x_in = torch::rand({150, 3});
  CANDOR::DistanceQueryer qdis(3);
  hnswIdx.insertTensor(x_in);
  cout << "insertion finish" << endl;
  size_t k = 1;
  auto ru = hnswIdx.searchTensor(x_in, k);
  for (int64_t i = 0; i < x_in.size(0); i++) {

    auto new_in = newTensor(x_in.slice(0, i, i + 1));
    cout << "looking for" << *new_in << endl;
    cout << endl << ru[i] << endl << endl;
  }

}
