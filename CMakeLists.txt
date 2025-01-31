cmake_minimum_required (VERSION 3.15)

# Options
option (DISABLE_SHADER_COMPILATION "Disable shader compilation" OFF)
option (USE_MINIMAL_DATA "Use minimal '_Data' (90MB)" OFF)
option (USE_DXC_FROM_PACKMAN_ON_AARCH64 "Use DXC from Packman for `aarch64`" ON)
set (SHADER_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/_Shaders")

# Cached
set (DXC_CUSTOM_PATH "custom/path/to/dxc" CACHE STRING "This DXC will be used if Vulkan SDK is not installed")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/_Bin" CACHE STRING "")
set (CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "")

# Create project
project (NRDSample LANGUAGES C CXX)

# Globals?
set_property (GLOBAL PROPERTY USE_FOLDERS ON)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_C_STANDARD 99)

if (MSVC)
    set (CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif ()

# Download dependencies using Packman
if (WIN32)
    set (PACKMAN_EXT ".cmd")
endif ()

if ((CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64") OR(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64"))
    if (WIN32)
        set (PACKMAN_PLATFORM "windows-x86_64")
    else ()
        set (PACKMAN_PLATFORM "linux-x86_64")
    endif ()
elseif ((CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64") OR(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64"))
    set (PACKMAN_PLATFORM "linux-aarch64")
else ()
    message (FATAL_ERROR "Unsupported architecture: '${CMAKE_SYSTEM_PROCESSOR}'")
endif ()

message ("Packman platform: ${PACKMAN_PLATFORM}")

if (USE_MINIMAL_DATA)
    set (NRI_DATA_VERSION 2.3_minimal)
    message ("Using minimal '_Data'")
else ()
    set (NRI_DATA_VERSION 2.3)
endif ()

execute_process (COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/External/Packman/packman${PACKMAN_EXT}" pull "${CMAKE_CURRENT_SOURCE_DIR}/Dependencies.xml" -p ${PACKMAN_PLATFORM} -t nri_data_version=${NRI_DATA_VERSION}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE PACKMAN_RESULT)

if (NOT "${PACKMAN_RESULT}" STREQUAL "0")
    message (FATAL_ERROR "Packman failed(code = ${PACKMAN_RESULT})")
endif ()

# Setup dependencies
if ((CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64") OR(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64"))
    if (("${DXC_CUSTOM_PATH}" STREQUAL "") AND USE_DXC_FROM_PACKMAN_ON_AARCH64)
        set (DXC_CUSTOM_PATH "${CMAKE_CURRENT_SOURCE_DIR}/External/DXC/bin/dxc" CACHE STRING "")
    endif ()
endif ()

# Compile options
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set (COMPILE_OPTIONS -msse4.1 -Wextra)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set (COMPILE_OPTIONS -msse4.1 -Wextra)
elseif (MSVC)
    set (COMPILE_OPTIONS /W4 /WX /wd4324 /wd4834)
else ()
    message (WARNING "Unknown compiler!")
endif ()

# Compile definitions
if (WIN32)
    set (COMPILE_DEFINITIONS WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS _UNICODE UNICODE)
endif ()

# External/NRIFramework
set (NRIF_SHADERS_PATH "${SHADER_OUTPUT_PATH}" CACHE STRING "")
add_subdirectory ("External/NRIFramework")

# External/NRD
set (NRD_DXC_CUSTOM_PATH ${DXC_CUSTOM_PATH} CACHE STRING "")
set (NRD_SHADERS_PATH "${SHADER_OUTPUT_PATH}" CACHE STRING "")
set (NRD_NORMAL_ENCODING "2" CACHE STRING "")
set (NRD_ROUGHNESS_ENCODING "1" CACHE STRING "")
option (NRD_DISABLE_SHADER_COMPILATION "" ${DISABLE_SHADER_COMPILATION})

add_subdirectory ("External/NRD")

# NRDSample
file (GLOB NRD_SAMPLE_SOURCE "Source/*.cpp")
source_group ("" FILES ${NRD_SAMPLE_SOURCE})

file (GLOB NRD_INTEGRATION_HEADERS "External/NRD/Integration/*.h" "External/NRD/Integration/*.hpp")
source_group ("NRD Integration" FILES ${NRD_INTEGRATION_HEADERS})

add_executable (${PROJECT_NAME} ${NRD_SAMPLE_SOURCE} ${NRD_INTEGRATION_HEADERS})

target_include_directories (${PROJECT_NAME} PRIVATE "Source" "External")
target_include_directories (${PROJECT_NAME} PRIVATE "External/NRIFramework/Include" "External/NRIFramework/Shaders")
target_include_directories (${PROJECT_NAME} PRIVATE "External/NRIFramework/External/NRI/Include")
target_include_directories (${PROJECT_NAME} PRIVATE "External/NRIFramework/External")
target_include_directories (${PROJECT_NAME} PRIVATE "External/NRD/Include" "External/NRD/Integration")
target_compile_definitions (${PROJECT_NAME} PRIVATE ${COMPILE_DEFINITIONS} PROJECT_NAME=${PROJECT_NAME} NRD_NORMAL_ENCODING=${NRD_NORMAL_ENCODING} NRD_ROUGHNESS_ENCODING=${NRD_ROUGHNESS_ENCODING} DXC_PATH=${DXC_PATH} DXC_SPIRV_PATH=${DXC_SPIRV_PATH})
target_compile_options (${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})

target_link_libraries (${PROJECT_NAME} PRIVATE NRIFramework NRI NRD)

if (UNIX)
    target_link_libraries (${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS} pthread X11)
endif ()

set_property (TARGET ${PROJECT_NAME} PROPERTY FOLDER "Sample")
set_property (TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_property (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

# Copy arguments for Visual Studio Smart Command Line Arguments extension
if (WIN32 AND MSVC)
    configure_file (.args "${CMAKE_BINARY_DIR}/${PROJECT_NAME}.args.json" COPYONLY)
endif ()

# Function - copy a library to the output folder of the project
function (copy_library PROJECT LIBRARY_NAME)
    add_custom_command (TARGET ${PROJECT} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBRARY_NAME} $<TARGET_FILE_DIR:${PROJECT}>
        COMMAND_EXPAND_LISTS)
endfunction ()

# DLSS
if (WIN32)
    find_library (NGX_D_LIB NAMES nvsdk_ngx_s_dbg.lib PATHS "External/NGX/lib/Windows_x86_64/x64")
    find_library (NGX_LIB NAMES nvsdk_ngx_s.lib PATHS "External/NGX/lib/Windows_x86_64/x64")
    find_file (DLSS_SR_DLL NAMES nvngx_dlss.dll PATHS "External/NGX/lib/Windows_x86_64/rel")
else ()
    find_library (NGX_D_LIB NAMES libnvsdk_ngx.a PATHS "External/NGX/lib/Linux_x86_64")
    find_library (NGX_LIB NAMES libnvsdk_ngx.a PATHS "External/NGX/lib/Linux_x86_64")
    find_file (DLSS_SR_DLL NAMES libnvidia-ngx-dlss.so.310.1.0 PATHS "External/NGX/lib/Linux_x86_64/rel")
endif ()

if (NGX_D_LIB AND NGX_LIB AND DLSS_SR_DLL)
    target_include_directories (${PROJECT_NAME} PRIVATE "$ENV{VULKAN_SDK}/include")
    target_link_libraries (${PROJECT_NAME} PRIVATE debug ${NGX_D_LIB})
    target_link_libraries (${PROJECT_NAME} PRIVATE optimized ${NGX_LIB})

    copy_library (${PROJECT_NAME} ${DLSS_SR_DLL})

    file (GLOB DLSS_INTEGRATION_HEADERS "Source/DLSS/*.h" "Source/DLSS/*.hpp")
    source_group ("DLSS Integration" FILES ${DLSS_INTEGRATION_HEADERS})
    target_sources (${PROJECT_NAME} PRIVATE ${DLSS_INTEGRATION_HEADERS})
else ()
    message (FATAL_ERROR "Can't find NGX LIB/DLL")
endif ()

# Shaders
file (GLOB_RECURSE SHADERS "Shaders/*.hlsl" "Shaders/*.hlsli" "External/NRD/External/MathLib/*.hlsli")
set_source_files_properties(${SHADERS} PROPERTIES VS_TOOL_OVERRIDE "None")

set (SHADERMAKE_GENERAL_ARGS
    --useAPI --binary --flatten --stripReflection --WX
    --sourceDir "Shaders"
    --shaderModel 6_6
    -c Shaders.cfg
    -o "${SHADER_OUTPUT_PATH}"
    -I "Shaders"
    -I "External"
    -I "External/NGX"
    -I "External/NRD/External"
    -I "External/NRIFramework/External/NRI/Include"
    -I "External/SHARC/Include"
    -D NRD_NORMAL_ENCODING=${NRD_NORMAL_ENCODING}
    -D NRD_ROUGHNESS_ENCODING=${NRD_ROUGHNESS_ENCODING}
    --sRegShift 100 --tRegShift 200 --bRegShift 300 --uRegShift 400
)

if (WIN32)
    add_custom_target (${PROJECT_NAME}_Shaders ALL
        COMMAND ShaderMake ${SHADERMAKE_GENERAL_ARGS} -p DXIL --compiler "${DXC_PATH}"
        COMMAND ShaderMake ${SHADERMAKE_GENERAL_ARGS} -p SPIRV --compiler "${DXC_SPIRV_PATH}" --hlsl2021
        DEPENDS ShaderMake
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VERBATIM
        SOURCES ${SHADERS}
    )
else ()
    add_custom_target (${PROJECT_NAME}_Shaders ALL
        COMMAND ShaderMake ${SHADERMAKE_GENERAL_ARGS} -p SPIRV --compiler "${DXC_SPIRV_PATH}" --hlsl2021
        DEPENDS ShaderMake
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VERBATIM
        SOURCES ${SHADERS}
    )
endif ()

set_property (TARGET ${PROJECT_NAME}_Shaders PROPERTY FOLDER "Sample")
add_dependencies (${PROJECT_NAME} ${PROJECT_NAME}_Shaders)
