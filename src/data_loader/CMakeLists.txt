add_sources(
        abstract_data_loader.cpp
        data_loader_table.cpp
        random_data_loader.cpp
        fvecs_data_loader.cpp
        zipf_data_loader.cpp
        exp_family_data_loader.cpp
        rbt_data_loader.cpp
)
if (ENABLE_HDF5)
    add_sources(hdf5_data_loader.cpp)
endif ()