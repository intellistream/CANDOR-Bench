/*! \file DataLoaderTable.h*/
//
// Created by tony on 10/05/23.
//

#ifndef CANDOR_INCLUDE_DataLOADER_DataLOADERTABLE_H_
#define CANDOR_INCLUDE_DataLOADER_DataLOADERTABLE_H_

#include <map>
#include <data_loader/abstract_data_loader.h>

namespace CANDOR {
/**
 * @ingroup CANDOR_DataLOADER
 * @{
 */
/**
 * @ingroup CANDOR_DataLOADER_Table The Table to index all Data loaders
 * @{
 */
/**
 * @class DataLoaderTable data_loader/data_loader_table.h
 * @brief The table class to index all Data loaders
 * @ingroup CANDOR_DataLOADER
 * @note  Default behavior
* - create
* - (optional) call @ref registerNewDataLoader for new loader
* - find a loader by @ref findDataLoader using its tag
 * @note default tags
 * - random @ref RandomDataLoader
 * - fvecs @ref FVECSDataLoader
 * - hdf5 @ref HDF5DataLoader
 * - zipf @ref ZipfDataLoader
 * - expFamily @ref ExpFamilyDataLoader
 * - exp, the exponential distribution in  @ref ExpFamilyDataLoader
 * - beta, the beta distribution in  @ref ExpFamilyDataLoader
 * - gaussian, the beta distribution in  @ref ExpFamilyDataLoader
 * - poisson, the poisson distribution in  @ref ExpFamilyDataLoader
 */
class DataLoaderTable {
 protected:
  std::map<std::string, CANDOR::AbstractDataLoaderPtr> loaderMap;
 public:
  /**
   * @brief The constructing function
   * @note  If new DataLoader wants to be included by default, please revise the following in *.cpp
   */
  DataLoaderTable();

  ~DataLoaderTable() {
  }

  /**
    * @brief To register a new loader
    * @param onew The new operator
    * @param tag THe name tag
    */
  void registerNewDataLoader(CANDOR::AbstractDataLoaderPtr dnew, std::string tag) {
    loaderMap[tag] = dnew;
  }

  /**
   * @brief find a dataloader in the table according to its name
   * @param name The nameTag of loader
   * @return The DataLoader, nullptr if not found
   */
  CANDOR::AbstractDataLoaderPtr findDataLoader(std::string name) {
    if (loaderMap.count(name)) {
      return loaderMap[name];
    }
    return nullptr;
  }

  /**
 * @ingroup CANDOR_DataLOADER_Table
 * @typedef DataLoaderTablePtr
 * @brief The class to describe a shared pointer to @ref DataLoaderTable

 */
  typedef std::shared_ptr<class CANDOR::DataLoaderTable> DataLoaderTablePtr;
/**
 * @ingroup CANDOR_DataLOADER_Table
 * @def newDataLoaderTable
 * @brief (Macro) To creat a new @ref  DataLoaderTable under shared pointer.
 */
#define newDataLoaderTable std::make_shared<CANDOR::DataLoaderTable>
};
/**
 * @}
 */
/**
 * @}
 */
} // CANDOR

#endif //INTELLISTREAM_INCLUDE_DataLOADER_DataLOADERTABLE_H_
