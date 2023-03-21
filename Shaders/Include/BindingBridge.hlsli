/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#if( defined( COMPILER_DXC ) )
    #if( defined( VULKAN ) )

        #define NRI_RESOURCE( resourceType, resourceName, regName, bindingIndex, setIndex ) \
            resourceType resourceName : register( regName ## bindingIndex, space ## setIndex )

        #define NRI_PUSH_CONSTANTS( structName, constantBufferName, bindingIndex ) \
            [[vk::push_constant]] structName constantBufferName

    #else

        #define NRI_RESOURCE( resourceType, resourceName, regName, bindingIndex, setIndex ) \
            resourceType resourceName : register( regName ## bindingIndex, space ## setIndex )

        #define NRI_PUSH_CONSTANTS( structName, constantBufferName, bindingIndex ) \
            ConstantBuffer<structName> constantBufferName : register( b ## bindingIndex, space0 )

    #endif
#else

    #define NRI_RESOURCE( resourceType, resourceName, regName, bindingIndex, setIndex ) \
        resourceType resourceName : register( regName ## bindingIndex )

    #define NRI_PUSH_CONSTANTS( structName, constantBufferName, bindingIndex ) \
        cbuffer structName ## _ ## constantBufferName : register( b ## bindingIndex ) \
        { \
            structName constantBufferName; \
        }

#endif
