/*! \file SPTAGIndex.h*/
//
// Created by tony on 04/01/24.
//

#ifndef CANDOR_INCLUDE_CANDOR_SPTAGIndex_H_
#define CANDOR_INCLUDE_CANDOR_SPTAGIndex_H_

#include <utils/abstract_c20_thread.hpp>
#include <utils/config_map.hpp>
#include <memory>
#include <vector>
#include <utils/intelli_tensor_op.hpp>
#include <faiss/IndexFlat.h>
#include <index/flat_index.h>
#include <SPTAG/AnnService/inc/Core/VectorIndex.h>
namespace CANDOR {

/**
 * @ingroup  CANDOR_lib_bottom The main body and interfaces of library function
 * @{
 */
/**
 * @class SPTAGIndex index/sptag_index.h
 * @brief The class of using SPTAG
 * @todo the revise and delete is not done yet
 * @note currently single thread by default
 * @note config parameters
 * - vecDim, the dimension of vectors, default 768, I64
 * - initialVolume, the initial volume of inline database tensor, default 1000, I64
 * - expandStep, the step of expanding inline database, default 100, I64
 * - SPTAGThreads, the number of involved threads, default 1, I64
 * - SPTAGNumberOfInitialDynamicPivots, Specifies the number of pivots used for partitioning the data into clusters during tree construction (relevant for BKT). Pivots are the points that the algorithm uses to split the data into clusters., DEFAULT 32, I64
 * - SPTAGMaxCheck,  The number of nodes to examine during a query. This affects the trade-off between query speed and accuracy. A higher value means more nodes are checked, resulting in better accuracy but slower queries., Default 8192. I64
 * - SPTAGGraphNeighborhoodSize,  Defines the size of the neighborhood graph during graph construction. This is used for neighbor search in the proximity graph. Default 32 I64
 * - SPTAGGraphNeighborhoodScale,   This parameter controls the scale of how the neighborhood size grows as the algorithm progresses through different stages of tree construction. Default 2.0, DOUBLE
 * - SPTAGRefineIterations, The number of iterations used during graph refinement. Refinement improves the quality of the nearest neighbor graph by updating the edges iteratively. dEFAULT 3, I64
 */
class SPTAGIndex : public FlatIndex {
 protected:
  std::shared_ptr<SPTAG::VectorIndex> sptag;
  int64_t SPTAGThreads = 1;
  bool isInitialized = true;
  int64_t SPTAGNumberOfInitialDynamicPivots,SPTAGMaxCheck,SPTAGGraphNeighborhoodSize,SPTAGRefineIterations;
  double SPTAGGraphNeighborhoodScale;
 public:
  SPTAGIndex() {

  }

  ~SPTAGIndex() {

  }

  /**
  * @brief load the initial tensors of a data base, use this BEFORE @ref insertTensor
  * @note This is majorly an offline function, and may be different from @ref insertTensor for some indexes
  * @param t the tensor, some index need to be single row
  * @return bool whether the loading is successful
  */
  virtual bool loadInitialTensor(torch::Tensor &t);
  /**
    * @brief reset this index to inited status
    */
  virtual void reset();
  /**
   * @brief set the index-specific config related to one index
   * @param cfg the config of this class
   * @return bool whether the configuration is successful
   */
  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

  /**
   * @brief insert a tensor
   * @param t the tensor, accept multiple rows
   * @return bool whether the insertion is successful
   */
  virtual bool insertTensor(torch::Tensor &t);

  /**
   * @brief search the k-NN of a query tensor, return the result tensors
   * @param t the tensor, allow multiple rows
   * @param k the returned neighbors
   * @return std::vector<torch::Tensor> the result tensor for each row of query
   */
  virtual std::vector<torch::Tensor> searchTensor(torch::Tensor &q, int64_t k);
  /**
   * @brief return the size of ingested tensors
   * @return
   */
  virtual int64_t size() {
    return lastNNZ + 1;
  }
};

/**
 * @ingroup  CANDOR_lib_bottom
 * @typedef SPTAGIndexPtr
 * @brief The class to describe a shared pointer to @ref  SPTAGIndex

 */
typedef std::shared_ptr<class CANDOR::SPTAGIndex> SPTAGIndexPtr;
/**
 * @ingroup  CANDOR_lib_bottom
 * @def newSPTAGIndex
 * @brief (Macro) To creat a new @ref  SPTAGIndex shared pointer.
 */
#define newSPTAGIndex std::make_shared<CANDOR::SPTAGIndex>
}
/**
 * @}
 */

#endif //INTELLISTREAM_INCLUDE_CPPALGOS_ABSTRACTCPPALGO_H_
