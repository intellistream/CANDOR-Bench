//
// Created by tony on 05/01/24.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <CANDY.h>
#include <iostream>
#include <CANDY/FlatAMMIPIndex.h>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDY;
TEST_CASE("Test SPTAG index", "[short]")
{
  //int a = 0;
  torch::manual_seed(114514);
  // place your test here
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  CANDY::IndexTable it;
  auto aknnIdx = it.getIndex("SPTAG");
  FlatAMMIPIndex bfIdx;
  cfg->edit("vecDim", (int64_t) 4);
  cfg->edit("metricType", "IP");
  aknnIdx->setConfig(cfg);
  bfIdx.setConfig(cfg);
  auto ta = torch::rand({4, 4});
  aknnIdx->loadInitialTensor(ta);
  bfIdx.loadInitialTensor(ta);
  auto tb = torch::rand({1, 4});
  std::cout <<"To insert:" <<tb<< std::endl;
  bfIdx.insertTensor(tb);
  aknnIdx->insertTensor(tb);
  auto ru2=bfIdx.searchTensor(tb,1);
  auto ru=aknnIdx->searchTensor(tb,1);
  std::cout <<"result form aknn:" <<ru[0]<< std::endl;
  std::cout <<"result form brutal force:" <<ru2[0]<< std::endl;
  //lshAPGIdx->insertTensor(tb);*/

}