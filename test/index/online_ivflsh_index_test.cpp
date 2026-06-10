//
// Created by tony on 05/01/24.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <index.h>
#include <iostream>
#include <index/online_pq_index/simple_stream_clustering.h>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDOR;

TEST_CASE("Test  online ivf lsh index insert", "[short]")
{
  int a = 0;
  torch::manual_seed(114514);
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  CANDOR::IndexTable it;
  auto onlineIVFLSHIdx = it.getIndex("onlineIVFLSH");
  cfg->edit("vecDim", (int64_t) 4);
  cfg->edit("encodeLen", (int64_t) 4);
  cfg->edit("candidateTimes", (int64_t) 4);
  // cfg->edit("numberOfBuckets", (int64_t) 2);
  onlineIVFLSHIdx->setConfig(cfg);
  auto db = torch::rand({6, 4});
  onlineIVFLSHIdx->insertTensor(db);
  std::cout << "data base is\n" << db << std::endl;
  auto query = db.slice(0, 2, 3);
  auto flatIndex = it.getIndex("flat");
  flatIndex->setConfig(cfg);
  flatIndex->insertTensor(db);
  auto flatRu = flatIndex->searchTensor(query, 2);
  std::cout << "flat result is\n" << flatRu[0] << std::endl;
  auto pqRu = onlineIVFLSHIdx->searchTensor(query, 1);
  std::cout << "IVFLSH result is\n" << pqRu[0] << std::endl;
  REQUIRE(a == 0);
}