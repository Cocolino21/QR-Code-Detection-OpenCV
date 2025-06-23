# Project definition
cmake_minimum_required(VERSION 3.21)
project(lab1)

# Search all directories that contain headers
macro(HEADER_DIRECTORIES return_list)
  file(GLOB_RECURSE new_list *.h)
  set(dir_list "")
  foreach(file_path ${new_list})
    get_filename_component(dir_path ${file_path} PATH)
    set(dir_list ${dir_list} ${dir_path})
  endforeach()
  list(REMOVE_DUPLICATES dir_list)
  set(${return_list} ${dir_list})
endmacro()

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE STRING "" FORCE)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "")
if(NOT CMAKE_BUILD_TYPE AND NOT (MSVC_IDE OR XCODE))
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type, one of: Release, Debug" FORCE)
endif()

add_definitions("-std=c++20")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -fcoroutines -fconcepts -fpermissive")

# Detect 64 or 32 bit
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  # 64-bit project
  SET(64_BIT_OS TRUE)
  message( STATUS "A 64-bit OS detected" )
else()
  SET(64_BIT_OS FALSE)
  message( STATUS "A 32-bit OS detected" )
endif()

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
   message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
   file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.16.1/conan.cmake"
                 "${CMAKE_BINARY_DIR}/conan.cmake")
endif()
include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_configure(REQUIRES spdlog/1.9.2 GENERATORS CMakeDeps)
conan_cmake_autodetect(settings)
conan_cmake_install(PATH_OR_REFERENCE . BUILD missing SETTINGS ${settings})

set(CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR} ${CMAKE_PREFIX_PATH})

# Find required packages
find_package(OpenCV REQUIRED)

# Find PkgConfig for QUIRC detection
find_package(PkgConfig)

# Print OpenCV information
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Try to find QUIRC library (multiple methods)
set(QUIRC_FOUND FALSE)
set(QUIRC_LIBRARIES "")
set(QUIRC_INCLUDE_DIRS "")

if(PKG_CONFIG_FOUND)
    # Method 1: Try pkg-config first
    pkg_check_modules(PC_QUIRC QUIET libquirc quirc)
    if(PC_QUIRC_FOUND)
        set(QUIRC_FOUND TRUE)
        set(QUIRC_LIBRARIES ${PC_QUIRC_LIBRARIES})
        set(QUIRC_INCLUDE_DIRS ${PC_QUIRC_INCLUDE_DIRS})
        message(STATUS "QUIRC found via pkg-config")
    endif()
endif()

if(NOT QUIRC_FOUND)
    # Method 2: Manual search for QUIRC (where you installed it)
    find_path(QUIRC_INCLUDE_DIR
        NAMES quirc.h
        PATHS
            /usr/local/include
            /usr/include
            /opt/local/include
        DOC "QUIRC include directory"
    )

    find_library(QUIRC_LIBRARY
        NAMES quirc libquirc
        PATHS
            /usr/local/lib
            /usr/lib
            /opt/local/lib
            /usr/lib/x86_64-linux-gnu
            /usr/lib/i386-linux-gnu
        DOC "QUIRC library"
    )

    if(QUIRC_INCLUDE_DIR AND QUIRC_LIBRARY)
        set(QUIRC_FOUND TRUE)
        set(QUIRC_LIBRARIES ${QUIRC_LIBRARY})
        set(QUIRC_INCLUDE_DIRS ${QUIRC_INCLUDE_DIR})
        message(STATUS "QUIRC found manually: ${QUIRC_LIBRARY}")
    endif()
endif()

# Display QUIRC status
if(QUIRC_FOUND)
    message(STATUS "QUIRC library status:")
    message(STATUS "    libraries: ${QUIRC_LIBRARIES}")
    message(STATUS "    include path: ${QUIRC_INCLUDE_DIRS}")
else()
    message(STATUS "QUIRC library: NOT FOUND")
    message(STATUS "    QR code decoding will be limited")
    message(STATUS "    Check if QUIRC was installed correctly")
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR})

# Get all *.c files
file(GLOB_RECURSE Sources RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cpp")
file(GLOB_RECURSE SBindings RELATIVE ${CMAKE_SOURCE_DIR} "bindings/*.cpp")

# Get all *.h files
file(GLOB_RECURSE Headers *.h)

if (DLIB_NO_GUI_SUPPORT)
  message("No GUI support, so we won't build the example.")
else()
  # Add everything to executable
  add_executable(${PROJECT_NAME} ${Headers} main.cpp ${Sources} ${SBindings}
          fileManager.cpp
          fileManager.hpp)

  # Link OpenCV (required)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

  # Link QUIRC if found
  if(QUIRC_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${QUIRC_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${QUIRC_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_QUIRC)
    message(STATUS "QUIRC will be linked to ${PROJECT_NAME}")
  else()
    message(STATUS "Building without QUIRC support")
  endif()

  # Get all header dirs to include
  header_directories(header_dir_list)

  list(LENGTH header_dir_list header_dir_list_count)
  message(STATUS "[INFO] Found ${header_dir_list_count} header directories.")
  file(COPY Images/ DESTINATION ${CMAKE_BINARY_DIR}/Images)
  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${header_dir_list}
  )

  # Packages
  # Fast and Reliable logging
  find_package(spdlog CONFIG REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC spdlog::spdlog)
endif()