
# DXC on Windows does not like forward slashes
if (WIN32)
    string(REPLACE "/" "\\" SHADER_INCLUDE_PATH "${SHADER_INCLUDE_PATH}")
    string(REPLACE "/" "\\" MATHLIB_INCLUDE_PATH "${MATHLIB_INCLUDE_PATH}")
    string(REPLACE "/" "\\" EXTERNAL_INCLUDE_PATH "${EXTERNAL_INCLUDE_PATH}")
endif()

# Find FXC and DXC
if (WIN32)
    if (DEFINED CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION)
        set (WINDOWS_SDK_VERSION ${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION})
    elseif (DEFINED ENV{WindowsSDKLibVersion})
        string (REGEX REPLACE "\\\\$" "" WINDOWS_SDK_VERSION "$ENV{WindowsSDKLibVersion}")
    else()
        message(FATAL_ERROR "WindowsSDK is not installed. (CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION is not defined; WindowsSDKLibVersion is '$ENV{WindowsSDKLibVersion}')")
    endif()

    get_filename_component(WINDOWS_SDK_ROOT
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;KitsRoot10]" ABSOLUTE)

    set(WINDOWS_SDK_BIN "${WINDOWS_SDK_ROOT}/bin/${WINDOWS_SDK_VERSION}/x64")

    # on Windows, FXC and DXC are part of WindowsSDK and there's also DXC in VulkanSDK which supports SPIR-V
    find_program(FXC_PATH "${WINDOWS_SDK_BIN}/fxc")
    if (NOT FXC_PATH)
        message(FATAL_ERROR "Can't find FXC: '${WINDOWS_SDK_BIN}/fxc'")
    endif()

    find_program(DXC_PATH "${WINDOWS_SDK_BIN}/dxc")
    if (NOT DXC_PATH)
        message(FATAL_ERROR "Can't find DXC: '${WINDOWS_SDK_BIN}/dxc'")
    endif()

    find_program(DXC_SPIRV_PATH "$ENV{VULKAN_SDK}/Bin/dxc")
    if (NOT DXC_SPIRV_PATH)
        message("Can't find VulkanSDK DXC: '$ENV{VULKAN_SDK}/Bin/dxc'")
        find_program(DXC_SPIRV_PATH "dxc" "${DXC_CUSTOM_PATH}")
        if (NOT DXC_SPIRV_PATH)
            message(FATAL_ERROR "Can't find DXC: Specify custom path using 'DXC_CUSTOM_PATH' parameter or install VulkanSDK")
        endif()
    endif()
else()
    # on Linux, VulkanSDK does not set VULKAN_SDK, but DXC can be called directly
    find_program(DXC_SPIRV_PATH "dxc")
    if (NOT DXC_SPIRV_PATH)
        find_program(DXC_SPIRV_PATH "${DXC_CUSTOM_PATH}")
        if (NOT DXC_SPIRV_PATH)
            message(FATAL_ERROR "Can't find DXC: VulkanSDK is not installed. Custom path can be specified using 'DXC_CUSTOM_PATH' parameter.")
        endif()
    endif()
endif()

message(STATUS "Using FXC path: '${FXC_PATH}'")
message(STATUS "Using DXC path: '${DXC_PATH}'")
message(STATUS "Using DXC (for SPIRV) path: '${DXC_SPIRV_PATH}'")

function(get_shader_profile_from_name FILE_NAME DXC_PROFILE FXC_PROFILE ENTRY_POINT)
    get_filename_component(EXTENSION ${FILE_NAME} EXT)
    if ("${EXTENSION}" STREQUAL ".cs.hlsl")
        set(DXC_PROFILE "cs_6_5" PARENT_SCOPE)
        set(FXC_PROFILE "cs_5_0" PARENT_SCOPE)
        set(ENTRY_POINT "-E main")
    endif()
    if ("${EXTENSION}" STREQUAL ".vs.hlsl")
        set(DXC_PROFILE "vs_6_5" PARENT_SCOPE)
        set(FXC_PROFILE "vs_5_0" PARENT_SCOPE)
        set(ENTRY_POINT "-E main")
    endif()
    if ("${EXTENSION}" STREQUAL ".hs.hlsl")
        set(DXC_PROFILE "hs_6_5" PARENT_SCOPE)
        set(FXC_PROFILE "hs_5_0" PARENT_SCOPE)
        set(ENTRY_POINT "-E main")
    endif()
    if ("${EXTENSION}" STREQUAL ".ds.hlsl")
        set(DXC_PROFILE "ds_6_5" PARENT_SCOPE)
        set(FXC_PROFILE "ds_5_0" PARENT_SCOPE)
    endif()
    if ("${EXTENSION}" STREQUAL ".gs.hlsl")
        set(DXC_PROFILE "gs_6_5" PARENT_SCOPE)
        set(FXC_PROFILE "gs_5_0" PARENT_SCOPE)
        set(ENTRY_POINT "-E main")
    endif()
    if ("${EXTENSION}" STREQUAL ".fs.hlsl")
        set(DXC_PROFILE "ps_6_5" PARENT_SCOPE)
        set(FXC_PROFILE "ps_5_0" PARENT_SCOPE)
        set(ENTRY_POINT "-E main")
    endif()
    if ("${EXTENSION}" STREQUAL ".rgen.hlsl")
        set(DXC_PROFILE "lib_6_5" PARENT_SCOPE)
    endif()
    if ("${EXTENSION}" STREQUAL ".rchit.hlsl")
        set(DXC_PROFILE "lib_6_5" PARENT_SCOPE)
    endif()
    if ("${EXTENSION}" STREQUAL ".rahit.hlsl")
        set(DXC_PROFILE "lib_6_5" PARENT_SCOPE)
    endif()
    if ("${EXTENSION}" STREQUAL ".rmiss.hlsl")
        set(DXC_PROFILE "lib_6_5" PARENT_SCOPE)
    endif()
endfunction()

macro(list_hlsl_headers HLSL_FILES HEADER_FILES)
    foreach(FILE_NAME ${HLSL_FILES})
        set(DXC_PROFILE "")
        set(FXC_PROFILE "")
        set(ENTRY_POINT "")
        get_shader_profile_from_name(${FILE_NAME} DXC_PROFILE FXC_PROFILE ENTRY_POINT)
        if ("${DXC_PROFILE}" STREQUAL "" AND "${FXC_PROFILE}" STREQUAL "")
            list(APPEND HEADER_FILES ${FILE_NAME})
            set_source_files_properties(${FILE_NAME} PROPERTIES VS_TOOL_OVERRIDE "None")
        endif()
    endforeach()
endmacro()

set (VK_S_SHIFT 100)
set (VK_T_SHIFT 200)
set (VK_B_SHIFT 300)
set (VK_U_SHIFT 400)
set (DXC_VK_SHIFTS
    -fvk-s-shift ${VK_S_SHIFT} 0 -fvk-s-shift ${VK_S_SHIFT} 1 -fvk-s-shift ${VK_S_SHIFT} 2
    -fvk-t-shift ${VK_T_SHIFT} 0 -fvk-t-shift ${VK_T_SHIFT} 1 -fvk-t-shift ${VK_T_SHIFT} 2
    -fvk-b-shift ${VK_B_SHIFT} 0 -fvk-b-shift ${VK_B_SHIFT} 1 -fvk-b-shift ${VK_B_SHIFT} 2
    -fvk-u-shift ${VK_U_SHIFT} 0 -fvk-u-shift ${VK_U_SHIFT} 1 -fvk-u-shift ${VK_U_SHIFT} 2)

macro(list_hlsl_shaders HLSL_FILES HEADER_FILES SHADER_FILES)
    foreach(FILE_NAME ${HLSL_FILES})
        get_filename_component(NAME_ONLY ${FILE_NAME} NAME)
        string(REGEX REPLACE "\\.[^.]*$" "" NAME_ONLY ${NAME_ONLY})
        string(REPLACE "." "_" BYTECODE_ARRAY_NAME "${NAME_ONLY}")
        set(DXC_PROFILE "")
        set(FXC_PROFILE "")
        set(ENTRY_POINT "")
        set(OUTPUT_PATH_DXBC "${SHADER_OUTPUT_PATH}/${NAME_ONLY}.dxbc")
        set(OUTPUT_PATH_DXIL "${SHADER_OUTPUT_PATH}/${NAME_ONLY}.dxil")
        set(OUTPUT_PATH_SPIRV "${SHADER_OUTPUT_PATH}/${NAME_ONLY}.spirv")
        get_shader_profile_from_name(${FILE_NAME} DXC_PROFILE FXC_PROFILE ENTRY_POINT)

        # add FXC compilation step (DXBC)
        if (NOT "${FXC_PROFILE}" STREQUAL "" AND NOT "${FXC_PATH}" STREQUAL "")
            add_custom_command(
                    OUTPUT ${OUTPUT_PATH_DXBC} ${OUTPUT_PATH_DXBC}.h
                    COMMAND ${FXC_PATH} /nologo ${ENTRY_POINT} /DCOMPILER_FXC=1 /T ${FXC_PROFILE}
                        /I "${EXTERNAL_INCLUDE_PATH}" /I "${SHADER_INCLUDE_PATH}" /I "${MATHLIB_INCLUDE_PATH}" /I "Include"
                        ${FILE_NAME} /Vn g_${BYTECODE_ARRAY_NAME}_dxbc /Fh ${OUTPUT_PATH_DXBC}.h /Fo ${OUTPUT_PATH_DXBC}
                        /WX /O3
                    MAIN_DEPENDENCY ${FILE_NAME}
                    DEPENDS ${HEADER_FILES}
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Source/Shaders"
                    VERBATIM
            )
            list(APPEND SHADER_FILES ${OUTPUT_PATH_DXBC})
        endif()
        # add DXC compilation step (DXIL)
        if (NOT "${DXC_PROFILE}" STREQUAL "" AND NOT "${DXC_PATH}" STREQUAL "")
            add_custom_command(
                    OUTPUT ${OUTPUT_PATH_DXIL} ${OUTPUT_PATH_DXIL}.h
                    COMMAND ${DXC_PATH} ${ENTRY_POINT} -DCOMPILER_DXC=1 -T ${DXC_PROFILE}
                        -I "${EXTERNAL_INCLUDE_PATH}" -I "${SHADER_INCLUDE_PATH}" -I "${MATHLIB_INCLUDE_PATH}" -I "Include"
                        ${FILE_NAME} -Vn g_${BYTECODE_ARRAY_NAME}_dxil -Fh ${OUTPUT_PATH_DXIL}.h -Fo ${OUTPUT_PATH_DXIL}
                        -WX -O3 -enable-16bit-types
                    MAIN_DEPENDENCY ${FILE_NAME}
                    DEPENDS ${HEADER_FILES}
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Source/Shaders"
                    VERBATIM
            )
            list(APPEND SHADER_FILES ${OUTPUT_PATH_DXIL})
        endif()
        # add one more DXC compilation step (SPIR-V)
        if (NOT "${DXC_PROFILE}" STREQUAL "" AND NOT "${DXC_SPIRV_PATH}" STREQUAL "")
            add_custom_command(
                    OUTPUT ${OUTPUT_PATH_SPIRV} ${OUTPUT_PATH_SPIRV}.h
                    COMMAND ${DXC_SPIRV_PATH} ${ENTRY_POINT} -DCOMPILER_DXC=1 -DVULKAN=1 -T ${DXC_PROFILE}
                        -I "${EXTERNAL_INCLUDE_PATH}" -I "${SHADER_INCLUDE_PATH}" -I "${MATHLIB_INCLUDE_PATH}" -I "Include"
                        ${FILE_NAME} -spirv -Vn g_${BYTECODE_ARRAY_NAME}_spirv -Fh ${OUTPUT_PATH_SPIRV}.h -Fo ${OUTPUT_PATH_SPIRV} ${DXC_VK_SHIFTS}
                        -WX -O3 -enable-16bit-types
                        -spirv -fspv-target-env=vulkan1.2 -fspv-extension=SPV_EXT_descriptor_indexing -fspv-extension=KHR
                    MAIN_DEPENDENCY ${FILE_NAME}
                    DEPENDS ${HEADER_FILES}
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Source/Shaders"
                    VERBATIM
            )
            list(APPEND SHADER_FILES ${OUTPUT_PATH_SPIRV})
        endif()
    endforeach()
endmacro()