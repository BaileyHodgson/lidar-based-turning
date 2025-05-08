cmake_minimum_required(VERSION 3.10)
project(RPLidarRotationDetector)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set GPIO and serial permissions
add_definitions(-D_REENTRANT)

# Add your source files
set(SOURCES
    main.cpp
    a1lidarrpi.cpp
)

# Add include directories
include_directories(
    ${CMAKE_SOURCE_DIR}
    /usr/local/include
)

# Link pigpio
link_libraries(pigpio pthread)

# Create the executable
add_executable(RPLidarRotationDetector ${SOURCES})

# Link required libraries
target_link_libraries(RPLidarRotationDetector
    pigpio
    pthread
    rt
)

# Optional: Show extra messages during configuration
message(STATUS "Project source dir: ${CMAKE_SOURCE_DIR}")
