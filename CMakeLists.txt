cmake_minimum_required(VERSION 3.15)

message("Processing 'NRD sample'...")

set(DXC_CUSTOM_PATH "custom/path/to/dxc" CACHE STRING "This DXC will be used if Vulkan SDK is not installed")

# Settings
option(DISABLE_SHADER_COMPILATION "disable compilation of shaders" OFF)
option(USE_MINIMAL_DATA "use minimal _Data (90MB)" OFF)
option(USE_DXC_FROM_PACKMAN_ON_AARCH64 "download DXC for aarch64 using Packman" ON)

project(Samples LANGUAGES C CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Detecting target architecture
if ((CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64"))
    set(PACKMAN_PLATFORM_SUFFIX "x86_64")
elseif ((CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64"))
    set(PACKMAN_PLATFORM_SUFFIX "aarch64")
else()
    message(FATAL_ERROR "Unknown architecture: '${CMAKE_SYSTEM_PROCESSOR}'")
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BUILD_ARCHITECTURE_SUFFIX "x86_64")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(BUILD_ARCHITECTURE_SUFFIX "x86")
endif()

# Use different scripts on different platforms
if (WIN32)
    set(PACKMAN_SCRIPT "packman.cmd")
    set(PACKMAN_PLATFORM "windows-${PACKMAN_PLATFORM_SUFFIX}")
else()
    set(PACKMAN_SCRIPT "packman")
    set(PACKMAN_PLATFORM "linux-${PACKMAN_PLATFORM_SUFFIX}")
endif()

# Choose data package
if (USE_MINIMAL_DATA)
    set(NRI_DATA_VERSION 2.0_minimal)
    message(STATUS "Using minimal _Data.")
else()
    set(NRI_DATA_VERSION 2.1)
endif()

# Verify that packman is available
find_file(PACKMAN NAMES ${PACKMAN_SCRIPT} PATHS "${CMAKE_CURRENT_SOURCE_DIR}/External/Packman")
if (NOT PACKMAN)
    message(FATAL_ERROR "Can't find packman: '${CMAKE_CURRENT_SOURCE_DIR}/External/Packman/${PACKMAN_SCRIPT}' (did you init submodules?)")
endif()

# Loading dependcies using packman
execute_process(COMMAND "${PACKMAN}" pull "${CMAKE_CURRENT_SOURCE_DIR}/Dependencies.xml" -t nri_data_version=${NRI_DATA_VERSION}
    -p ${PACKMAN_PLATFORM} RESULT_VARIABLE PACKMAN_RESULT)
if ("${PACKMAN_RESULT}" STREQUAL "0")
    message(STATUS "Packman result: success (platform: '${PACKMAN_PLATFORM}')")
else()
    message(FATAL_ERROR "Packman result: ${PACKMAN_RESULT} (platform: '${PACKMAN_PLATFORM}')")
endif()

# Packman downloads some dependencies on aarch64
if ((CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64"))
    if (("${DXC_CUSTOM_PATH}" STREQUAL "") AND USE_DXC_FROM_PACKMAN_ON_AARCH64)
        set(DXC_CUSTOM_PATH "${CMAKE_CURRENT_SOURCE_DIR}/External/DXC/bin/dxc" CACHE STRING)
    endif()
endif()

# Shader compilation
if (NOT DISABLE_SHADER_COMPILATION)
    # Include path for DXC and FXC (used in ShaderCompilation.cmake)
    set(SHADER_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Source/Shaders")
    set(MATHLIB_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/External/NRIFramework/External/MathLib")
    set(EXTERNAL_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/External")
    set(NRD_SHADER_HEADER_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/External/NRD/Include")
    set(SHADER_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/_Data/Shaders")

    # List HLSL shaders and headers
    file(GLOB_RECURSE HLSL_FILES "Source/Shaders/*.hlsl" "Source/Shaders/*.hlsli")
    file(GLOB_RECURSE HLSL_MATHLIB_HEADER_FILES "${MATHLIB_INCLUDE_PATH}/*.hlsli")
    file(GLOB_RECURSE HLSL_NRD_HEADER_FILES "${NRD_SHADER_HEADER_INCLUDE_PATH}/*.hlsli")

    include(ShaderCompilation.cmake)

    # Create "_Data/Shaders"
    add_custom_target(SampleCreateFolderForShaders ALL COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_PATH})
    set_property(TARGET SampleCreateFolderForShaders PROPERTY FOLDER "Shaders")

    # Define build rules for the shaders
    list_hlsl_headers("${HLSL_FILES}" HEADER_FILES)
    list_hlsl_headers("${HLSL_MATHLIB_HEADER_FILES}" HEADER_FILES)
    list_hlsl_headers("${HLSL_NRD_HEADER_FILES}" HEADER_FILES)
    list_hlsl_shaders("${HLSL_FILES}" "${HEADER_FILES}" SHADER_FILES)
    add_custom_target(SampleShaders ALL DEPENDS ${SHADER_FILES} SOURCES "${HEADER_FILES}")
    add_dependencies(SampleShaders SampleCreateFolderForShaders)
    set_property(TARGET SampleShaders PROPERTY FOLDER "Shaders")
else()
    option(NRD_DISABLE_SHADER_COMPILATION "" ON)
endif()

# MathLib requires SSE4.1
if ((CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64"))
    set(ARCHITECTURE_COMPILE_OPTIONS -msse4.1)
endif()

# Compiler-specific options
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # -Wno-pragma-pack: assimp causes a number of these warnings
    # warning: the current #pragma pack alignment value is modified in the included file [-Wpragma-pack]
    add_compile_options(${ARCHITECTURE_COMPILE_OPTIONS} -Wno-missing-braces -Wno-return-type-c-linkage -Wno-pragma-pack)
    set(DETEX_COMPILE_OPTIONS -Wno-unused-function)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(${ARCHITECTURE_COMPILE_OPTIONS})
endif()

if (WIN32)
    # On Windows, GLFW requires /MT
    add_compile_options($<$<CONFIG:Debug>:/MTd> $<$<CONFIG:Release>:/MT> $<$<CONFIG:RelWithDebInfo>:/MT> $<$<CONFIG:MinSizeRel>:/MT>)
endif()

set(OUTPUT_BIN_PATH "${CMAKE_SOURCE_DIR}/_Build")

# NRI
set(NRI_PROJECT_FOLDER "External/NRI")
set(NRI_OUTPUT_BIN_PATH "${OUTPUT_BIN_PATH}")
add_subdirectory("External/NRIFramework/External/NRI")

# NRIFramework
set(NRIF_OUTPUT_PATH "${OUTPUT_BIN_PATH}")
add_subdirectory("External/NRIFramework")

# NRD
set(NRD_DXC_CUSTOM_PATH ${DXC_CUSTOM_PATH})
set(NRD_SHADER_OUTPUT_PATH "${SHADER_OUTPUT_PATH}" CACHE STRING "")
set(NRD_PROJECT_FOLDER "External/NRD")
set(NRD_OUTPUT_BIN_PATH "${OUTPUT_BIN_PATH}")
add_subdirectory("External/NRD")

if(MSVC)
    set(PLATFORM_DEFINITIONS WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS _UNICODE UNICODE)
endif()

# Function copies a library to the output folder of the project (1 LOC instead of 3LOC)
function(copy_library PROJECT LIBRARY_NAME)
    add_custom_command(TARGET ${PROJECT} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBRARY_NAME} $<TARGET_FILE_DIR:${PROJECT}>
        COMMAND_EXPAND_LISTS)
endfunction()

# Creates a project for sample and adds dependent operations

file(GLOB NRD_SAMPLE_SOURCE "Source/*.cpp")
source_group("" FILES ${NRD_SAMPLE_SOURCE})

file(GLOB NRD_INTEGRATION_HEADERS "External/NRD/Integration/*.h" "External/NRD/Integration/*.hpp")
source_group("NRD Integration" FILES ${NRD_INTEGRATION_HEADERS})

add_executable(NRDSample ${NRD_SAMPLE_SOURCE} ${NRD_INTEGRATION_HEADERS})
target_include_directories(NRDSample PRIVATE "Source" "External")
target_include_directories(NRDSample PRIVATE "External/NRIFramework/Include")
target_include_directories(NRDSample PRIVATE "External/NRIFramework/External/NRI/Include")
target_include_directories(NRDSample PRIVATE "External/NRIFramework/External")
target_include_directories(NRDSample PRIVATE "External/NRD/Include" "External/NRD/Integration")
target_compile_definitions(NRDSample PRIVATE ${PLATFORM_DEFINITIONS} PROJECT_NAME=NRDSample)

if (NOT DISABLE_SHADER_COMPILATION)
    add_dependencies(NRDSample SampleShaders)
endif()

target_link_libraries(NRDSample PRIVATE NRIFramework NRI NRD)
if (UNIX)
    target_link_libraries(NRDSample PRIVATE ${CMAKE_DL_LIBS} pthread X11)
endif()

set_target_properties(NRDSample PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_BIN_PATH}/$<CONFIG>")
set_property(TARGET NRDSample PROPERTY FOLDER "Samples")
set_property(TARGET NRDSample PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT NRDSample)

# Copy arguments for Visual Studio Smart Command Line Arguments extension
if (WIN32 AND MSVC)
    configure_file(.args "${CMAKE_BINARY_DIR}/NRDSample.args.json" COPYONLY)
endif()

# DLSS
if (WIN32)
    find_library(INPUT_NGX_DEBUG_LIB NAMES nvsdk_ngx_s_dbg.lib PATHS "External/NGX/lib/Windows_x86_64/x86_64")
    find_library(INPUT_NGX_RELEASE_LIB NAMES nvsdk_ngx_s.lib PATHS "External/NGX/lib/Windows_x86_64/x86_64")
    find_file(INPUT_NGX_DLL NAMES nvngx_dlss.dll PATHS "External/NGX/lib/Windows_x86_64/rel")
else()
    find_library(INPUT_NGX_DEBUG_LIB NAMES libnvsdk_ngx.a PATHS "External/NGX/lib/Linux_x86_64")
    find_library(INPUT_NGX_RELEASE_LIB NAMES libnvsdk_ngx.a PATHS "External/NGX/lib/Linux_x86_64")
    find_file(INPUT_NGX_DLL NAMES libnvidia-ngx-dlss.so.2.3.1 PATHS "External/NGX/lib/Linux_x86_64/rel")
endif()

if ((NOT INPUT_NGX_DEBUG_LIB) OR (NOT INPUT_NGX_RELEASE_LIB))
    message(WARNING "Can't find NGX LIB.")
endif()

if (NOT INPUT_NGX_DLL)
    message(WARNING "Can't find NGX DLL.")
endif()

if (INPUT_NGX_DEBUG_LIB AND INPUT_NGX_RELEASE_LIB AND INPUT_NGX_DLL)
    target_include_directories(NRDSample PRIVATE "External/NRIFramework/External/NRI/External/vulkan/Include")
    target_link_libraries(NRDSample PRIVATE debug ${INPUT_NGX_DEBUG_LIB})
    target_link_libraries(NRDSample PRIVATE optimized ${INPUT_NGX_RELEASE_LIB})
    copy_library(NRDSample ${INPUT_NGX_DLL})

    file(GLOB DLSS_INTEGRATION_HEADERS "Source/DLSS/*.h" "Source/DLSS/*.hpp")
    source_group("DLSS Integration" FILES ${DLSS_INTEGRATION_HEADERS})
    target_sources(NRDSample PRIVATE ${DLSS_INTEGRATION_HEADERS})
endif()

# Copy NRD tests to _Data, but only if they do not exist in _Data
file(GLOB NRD_TESTS "Tests/*")
foreach(TEST ${NRD_TESTS})
    get_filename_component(TEST_FILENAME ${TEST} NAME)
    set(DEST_TEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/_Data/Tests/${TEST_FILENAME}")
    if (NOT EXISTS "${DEST_TEST_PATH}")
        configure_file(${TEST} "${DEST_TEST_PATH}" COPYONLY)
    endif()
endforeach()
