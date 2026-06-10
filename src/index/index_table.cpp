//
// Created by tony on 25/05/23.
//

#include <index/bucketed_flat_index.h>
#include <index/buffered_congestion_drop_index.h>
#include <index/congestion_drop_index.h>
#include <index/dpg_index.h>
#include <index/faiss_index.h>
// #include <index/flann_index.h>
//#include <index/flat_ammip_index.h>
//#include <index/flat_ammip_obj_index.h>
#include <index/flat_index.h>
// #include <index/hnsw_naive_index.h>
#include <index/index_table.h>
#include <index/lshapg_index.h>
#include <index/nn_descent_index.h>
//#include <index/online_ivfl2_h_index.h>
//#include <index/online_ivflsh_index.h>
#include <index/online_pq_index.h>
//#include <index/pq_index.h>
#include <index/parallel_partition_index.h>
//#include <index/yin_yang_graph_index.h>
// #include <index/flat_gpu_index.h>
//#include <index/yin_yang_graph_simple_index.h>
#include <include/opencl_config.h>
#include <include/ray_config.h>
#include <include/sptag_config.h>
#if CANDOR_CL == 1
//#include <CPPAlgos/CLMMCPPAlgo.h>
#endif
#if CANDOR_RAY == 1
#include <index/distributed_partition_index.h>
#endif
#if CANDOR_SPTAG == 1
#include <index/sptag_index.h>
#endif
//#ifdef ENABLE_CUDA
// #include <index/song/song.hpp>
//#endif
namespace CANDOR {
CANDOR::IndexTable::IndexTable() {
  indexMap["null"] = newAbstractIndex();
  indexMap["flat"] = newFlatIndex();
  //indexMap["flatAMMIP"] = newFlatAMMIPIndex();
  //indexMap["flatAMMIPObj"] = newFlatAMMIPObjIndex();
  indexMap["bucketedFlat"] = newBucketedFlatIndex();
  //indexMap["parallelPartition"] = newParallelPartitionIndex();
  indexMap["onlinePQ"] = newOnlinePQIndex();
  //indexMap["onlineIVFLSH"] = newOnlineIVFLSHIndex();
  //indexMap["onlineIVFL2H"] = newOnlineIVFL2HIndex();
  //indexMap["PQ"] = newPQIndex();
  // indexMap["HNSWNaive"] = newHNSWNaiveIndex();
  // indexMap["NSW"] = newNSWIndex();
  indexMap["faiss"] = newFaissIndex();
  //indexMap["yinYang"] = newYinYangGraphIndex();
  //indexMap["yinYangSimple"] = newYinYangGraphSimpleIndex();
  indexMap["congestionDrop"] = newCongestionDropIndex();
  indexMap["bufferedCongestionDrop"] = newBufferedCongestionDropIndex();
  indexMap["nnDescent"] = newNNDescentIndex();
  // indexMap["Flann"] = newFlannIndex();
  indexMap["DPG"] = newDPGIndex();
  indexMap["LSHAPG"] = newLSHAPGIndex();
  // indexMap["flatGPU"] = newFlatGPUIndex();
//#ifdef ENABLE_CUDA
  // indexMap["SONG"] = newSONG();
//#endif
#if CANDOR_CL == 1
  // indexMap["cl"] = newCLMMCPPAlgo();
#endif
#if CANDOR_RAY == 1
  indexMap["distributedPartition"] = newDistributedPartitionIndex();
#endif
#if CANDOR_SPTAG == 1
  indexMap["SPTAG"] = newSPTAGIndex();
#endif
}
}  // namespace CANDOR
