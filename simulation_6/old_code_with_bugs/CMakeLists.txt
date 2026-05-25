cmake_minimum_required(VERSION 3.14)
project(BacterialPhageSim)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add all source files
set(SOURCES
    main_6.cpp
    bacteria_6.cpp
    simulation_grid_6.cpp
    phage_6.cpp
    diffusion_6.cpp     # Add this if/when implemented
)

# Header include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/external)

# JSON library (assumes header-only version is available in external/)
include_directories(${CMAKE_SOURCE_DIR}/external)

# Define executable and link JSON
add_executable(sim ${SOURCES})

