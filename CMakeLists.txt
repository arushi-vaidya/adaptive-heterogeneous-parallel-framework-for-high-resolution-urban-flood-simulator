cmake_minimum_required(VERSION 3.16)
project(urban_flood_hpc VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FLOOD_ENABLE_OPENMP "Build the OpenMP solver backend when a runtime is available" ON)
option(FLOOD_ENABLE_MPI "Probe for an optional MPI toolchain" ON)
set(FLOOD_NUMERICAL_TOLERANCE "1e-8" CACHE STRING "Absolute depth tolerance for backend comparisons, in metres")
set(FLOOD_MOMENTUM_TOLERANCE "1e-8" CACHE STRING "Absolute momentum tolerance for backend comparisons, in m^2/s")

if(FLOOD_ENABLE_MPI)
    find_package(MPI QUIET COMPONENTS CXX)
    if(MPI_CXX_FOUND)
        set(FLOOD_MPI_TOOLCHAIN_FOUND TRUE)
        if(MPIEXEC_EXECUTABLE)
            set(FLOOD_MPI_RUNTIME_FOUND TRUE)
        endif()
        message(STATUS "MPI C++ toolchain and launcher detected")
    else()
        message(STATUS "MPI unavailable: implementation source exists, but no MPI C++ compiler/runtime was detected")
    endif()
endif()

add_library(flood_core
    src/grid/Grid.cpp
    src/grid/AdaptiveGrid.cpp
    src/physics/Rainfall.cpp
    src/solver/SerialSolver.cpp
    src/solver/AdaptiveSolver.cpp
    src/simulation/Scenarios.cpp
    src/io/Output.cpp
    src/io/CsvInput.cpp
)
target_include_directories(flood_core PUBLIC include)
target_compile_options(flood_core PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(flood_core PUBLIC FLOOD_MPI_SOURCE_AVAILABLE=1)

if(FLOOD_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(flood_core PUBLIC OpenMP::OpenMP_CXX)
        set(FLOOD_OPENMP_FOUND TRUE)
    elseif(APPLE)
        find_path(FLOOD_LIBOMP_INCLUDE_DIR omp.h HINTS /opt/homebrew/opt/libomp/include)
        find_library(FLOOD_LIBOMP_LIBRARY omp HINTS /opt/homebrew/opt/libomp/lib)
        if(FLOOD_LIBOMP_INCLUDE_DIR AND FLOOD_LIBOMP_LIBRARY)
            target_include_directories(flood_core PUBLIC ${FLOOD_LIBOMP_INCLUDE_DIR})
            target_compile_options(flood_core PUBLIC -Xpreprocessor -fopenmp)
            target_link_libraries(flood_core PUBLIC ${FLOOD_LIBOMP_LIBRARY})
            set(FLOOD_OPENMP_FOUND TRUE)
        endif()
    endif()
    if(FLOOD_OPENMP_FOUND)
        target_sources(flood_core PRIVATE src/openmp/OpenMPSolver.cpp)
        target_sources(flood_core PRIVATE src/openmp/AdaptiveOpenMPSolver.cpp)
        target_compile_definitions(flood_core PUBLIC FLOOD_HAS_OPENMP=1)
    else()
        message(WARNING "OpenMP runtime not found; only the serial backend will be built")
    endif()
endif()

if(FLOOD_MPI_TOOLCHAIN_FOUND)
    target_compile_definitions(flood_core PUBLIC FLOOD_MPI_TOOLCHAIN_FOUND=1)
endif()
if(FLOOD_MPI_RUNTIME_FOUND)
    target_compile_definitions(flood_core PUBLIC FLOOD_MPI_RUNTIME_FOUND=1)
endif()

if(FLOOD_MPI_TOOLCHAIN_FOUND)
    target_sources(flood_core PRIVATE src/mpi/MpiSolver.cpp)
    target_link_libraries(flood_core PUBLIC MPI::MPI_CXX)
    target_compile_definitions(flood_core PUBLIC FLOOD_HAS_MPI=1)
endif()

add_executable(flood_sim src/main.cpp)
target_link_libraries(flood_sim PRIVATE flood_core)

include(CTest)
if(BUILD_TESTING)
    set(FLOOD_MASS_TOLERANCE "1e-8" CACHE STRING "Absolute mass-balance tolerance in cubic metres for regression tests")
    add_executable(flood_tests tests/test_solver.cpp)
    target_link_libraries(flood_tests PRIVATE flood_core)
    target_compile_definitions(flood_tests PRIVATE FLOOD_MASS_TOLERANCE=${FLOOD_MASS_TOLERANCE})
    add_test(NAME flood_tests COMMAND flood_tests)
    set_tests_properties(flood_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_executable(flood_adaptive_grid_tests tests/test_adaptive_grid.cpp)
    target_link_libraries(flood_adaptive_grid_tests PRIVATE flood_core)
    add_test(NAME flood_adaptive_grid_tests COMMAND flood_adaptive_grid_tests)
    add_executable(flood_adaptive_solver_tests tests/test_adaptive_solver.cpp)
    target_link_libraries(flood_adaptive_solver_tests PRIVATE flood_core)
    target_compile_definitions(flood_adaptive_solver_tests PRIVATE
        FLOOD_MASS_TOLERANCE=${FLOOD_MASS_TOLERANCE})
    add_test(NAME flood_adaptive_solver_tests COMMAND flood_adaptive_solver_tests)
    add_test(NAME flood_config_smoke
        COMMAND flood_sim --config ${CMAKE_CURRENT_SOURCE_DIR}/config/example.ini
                --output ${CMAKE_CURRENT_BINARY_DIR}/config-smoke-output)
    add_executable(flood_capability_tests tests/test_capabilities.cpp)
    target_link_libraries(flood_capability_tests PRIVATE flood_core)
    add_test(NAME flood_capability_tests COMMAND flood_capability_tests)
    if(FLOOD_OPENMP_FOUND)
        add_executable(flood_backend_tests tests/test_backend_consistency.cpp)
        target_link_libraries(flood_backend_tests PRIVATE flood_core)
        target_compile_definitions(flood_backend_tests PRIVATE
            FLOOD_NUMERICAL_TOLERANCE=${FLOOD_NUMERICAL_TOLERANCE}
            FLOOD_MOMENTUM_TOLERANCE=${FLOOD_MOMENTUM_TOLERANCE})
        add_test(NAME flood_backend_tests COMMAND flood_backend_tests)
        set_tests_properties(flood_backend_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        add_executable(flood_adaptive_openmp_tests tests/test_adaptive_openmp_solver.cpp)
        target_link_libraries(flood_adaptive_openmp_tests PRIVATE flood_core)
        target_compile_definitions(flood_adaptive_openmp_tests PRIVATE
            FLOOD_MASS_TOLERANCE=${FLOOD_MASS_TOLERANCE})
        add_test(NAME flood_adaptive_openmp_tests COMMAND flood_adaptive_openmp_tests)
        set_tests_properties(flood_adaptive_openmp_tests PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
    if(FLOOD_MPI_TOOLCHAIN_FOUND AND MPIEXEC_EXECUTABLE)
        add_executable(flood_mpi_tests tests/test_mpi_consistency.cpp)
        target_link_libraries(flood_mpi_tests PRIVATE flood_core MPI::MPI_CXX)
        target_compile_definitions(flood_mpi_tests PRIVATE
            FLOOD_NUMERICAL_TOLERANCE=${FLOOD_NUMERICAL_TOLERANCE}
            FLOOD_MOMENTUM_TOLERANCE=${FLOOD_MOMENTUM_TOLERANCE}
            FLOOD_MASS_TOLERANCE=${FLOOD_MASS_TOLERANCE})
        foreach(process_count 1 2 3 4 6)
            foreach(scenario flat-basin slope dam-break rain-drain wet-dry lake-at-rest)
                add_test(NAME flood_mpi_${scenario}_${process_count}
                    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${process_count}
                            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:flood_mpi_tests> ${scenario}
                            ${MPIEXEC_POSTFLAGS})
                set_tests_properties(flood_mpi_${scenario}_${process_count} PROPERTIES
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} TIMEOUT 120)
            endforeach()
        endforeach()
    endif()
endif()