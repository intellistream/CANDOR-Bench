add_subdirectory(utils)

add_subdirectory(index)
#add_subdirectory(data_loader)
#gcc options
if (DEFINED ENV{CANDOR_CL})
    add_sources(
            cl/cl_container.cpp
    )
endif ()
add_sources(
        #bindings/torch_bounds.cpp
)
include_directories("../include/")
