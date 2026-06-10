/*! \file FVECSDataLoader.h*/
//
// Created by tony on 10/05/23.
//

#ifndef CANDOR_INCLUDE_DATALOADER_FVECSDataLoader_H_
#define CANDOR_INCLUDE_DATALOADER_FVECSDataLoader_H_

#include <utils/config_map.hpp>
#include <utils/intelli_tensor_op.hpp>
#include <assert.h>
//#include <torch/torch.h>
#include <memory>
#include <data_loader/abstract_data_loader.h>
namespace CANDOR {
/**
 * @ingroup CANDOR_DataLOADER
 * @{
 */
/**
 * @ingroup CANDOR_DataLOADER_FVECS The dataloader for *.vecs file
 * @{
 */
/**
 * @class FVECSDataLoader data_loader/fvecs_data_loader.h
 * @brief The class for loading *.fvecs data
 * @ingroup CANDOR_DataLOADER
 * @note:
 * - Must have a global config by @ref setConfig
 * @note  Default behavior
* - create
* - call @ref setConfig, this function will also generate the tensor A and B correspondingly
* - call @ref getData to get the raw data
* - call  @ref getQuery to get the query
* @note parameters of config
* - vecDim, the dimension of vectors, default 128, I64
* - vecVolume, the volume of vectors, default 10000, I64
* - dataPath, the path to the data file, datasets/fvecs/sift10K/siftsmall_base.fvecs, String
* - normalizeTensor, whether or not normalize the tensors in L2, 1 (yes), I64
* - useSeparateQuery, whether or not load query separately, 1, I64
* - queryPath, the path to query file, datasets/fvecs/sift10K/siftsmall_query.fvecs. String
* - queryNoiseFraction, the fraction of noise in query, default 0, allow 0~1, Double
 * - no effect when query is loaded from separate file
* - querySize, the size of query, default 10, I64
* - seed, the random seed, default 7758258, I64
*  @note: default name tags
* - "fvecs": @ref FVECSDataLoader
 */
class FVECSDataLoader : public AbstractDataLoader {
 protected:
  torch::Tensor A, B;
  int64_t vecDim, vecVolume, querySize, seed;
  int64_t normalizeTensor;
  double queryNoiseFraction;
  int64_t useSeparateQuery;
  bool generateData(std::string fname);
  bool generateQuery(std::string fname);

 public:
  FVECSDataLoader() = default;

  ~FVECSDataLoader() = default;

  /**
     * @brief Set the GLOBAL config map related to this loader
     * @param cfg The config map
      * @return bool whether the config is successfully set
      * @note
     */
  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

  /**
   * @brief get the data tensor
   * @return the generated data tensor
   */
  virtual torch::Tensor getData();

  /**
  * @brief get the query tensor
  * @return the generated query tensor
  */
  virtual torch::Tensor getQuery();
  /**
   * @brief the inline function to load tensor from fvecs file
   * @param fname the name of file
   * @return the genearetd tensor
   */
  static torch::Tensor tensorFromFVECS(std::string fname);
};

/**
 * @ingroup CANDOR_MatrixLOADER_FVECS
 * @typedef FVECSDataLoaderPtr
 * @brief The class to describe a shared pointer to @ref FVECSDataLoader

 */
typedef std::shared_ptr<class CANDOR::FVECSDataLoader> FVECSDataLoaderPtr;
/**
 * @ingroup CANDOR_MatrixLOADER_FVECS
 * @def newFVECSDataLoader
 * @brief (Macro) To creat a new @ref FVECSDataLoader under shared pointer.
 */
#define newFVECSDataLoader std::make_shared<CANDOR::FVECSDataLoader>
/**
 * @}
 */
/**
 * @}
 */
} // CANDOR

#endif //CANDOR_INCLUDE_MATRIXLOADER_FVECSDataLoader_H_
