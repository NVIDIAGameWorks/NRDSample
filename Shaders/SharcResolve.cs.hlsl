/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Include/Shared.hlsli"
#include "Include/RaytracingShared.hlsli"

#include "SharcCommon.h"

[numthreads( LINEAR_BLOCK_SIZE, 1, 1 )]
void main( uint threadIndex : SV_DispatchThreadID )
{
    GridParameters gridParameters = ( GridParameters )0;
    gridParameters.cameraPosition = gCameraGlobalPos.xyz;
    gridParameters.cameraPositionPrev = gCameraGlobalPosPrev.xyz;
    gridParameters.sceneScale = SHARC_SCENE_SCALE;
    gridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;

    HashMapData hashMapData;
    hashMapData.capacity = SHARC_CAPACITY;
    hashMapData.hashEntriesBuffer = gInOut_SharcHashEntriesBuffer;

    SharcResolveEntry( threadIndex, gridParameters, hashMapData, gInOut_SharcHashCopyOffsetBuffer, gInOut_SharcVoxelDataBuffer, gInOut_SharcVoxelDataBufferPrev, gSharcMaxAccumulatedFrameNum, SHARC_STALE_FRAME_NUM_MIN );
}
