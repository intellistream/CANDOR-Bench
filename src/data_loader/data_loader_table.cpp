//
// Created by tony on 10/05/23.
//

#include <data_loader/data_loader_table.h>
#include <data_loader/random_data_loader.h>
#include <data_loader/zipf_data_loader.h>
#include <data_loader/exp_family_data_loader.h>
#include <data_loader/fvecs_data_loader.h>
#include <data_loader/rbt_data_loader.h>
#include <include/hdf5_config.h>
#if CANDOR_HDF5 == 1
#include <data_loader/hdf5_data_loader.h>
#endif

namespace CANDOR {
static CANDOR::AbstractDataLoaderPtr genExpFamilyLoader(INTELLI::ConfigMapPtr cfgHijack, std::string tag) {
  auto expLd = newExpFamilyDataLoader();
  cfgHijack->edit("distributionOverwrite", tag);
  expLd->hijackConfig(cfgHijack);
  return expLd;
}
/**
 * @note revise me if you need new loader
 */
CANDOR::DataLoaderTable::DataLoaderTable() {
  loaderMap["null"] = newAbstractDataLoader();
  loaderMap["random"] = newRandomDataLoader();
  loaderMap["fvecs"] = newFVECSDataLoader();
  loaderMap["zipf"] = newZipfDataLoader();
  loaderMap["expFamily"] = newExpFamilyDataLoader();
  /**
   * @brief more specific loader oin exp family
   */
  INTELLI::ConfigMapPtr cfgHijack = newConfigMap();
  loaderMap["exp"] = genExpFamilyLoader(cfgHijack, "exp");
  loaderMap["beta"] = genExpFamilyLoader(cfgHijack, "beta");
  loaderMap["gaussian"] = genExpFamilyLoader(cfgHijack, "gaussian");
  loaderMap["poisson"] = genExpFamilyLoader(cfgHijack, "poisson");
  loaderMap["rbt"] = newRBTDataLoader();
#if CANDOR_HDF5 == 1
  loaderMap["hdf5"] = newHDF5DataLoader();
#endif
}

} // CANDOR