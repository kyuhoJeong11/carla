// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "VehicleControl.generated.h"

USTRUCT(BlueprintType)
struct CARLA_API FVehicleControl
{
  GENERATED_BODY()

  UPROPERTY(Category = "Vehicle Control", EditAnywhere, BlueprintReadWrite)
  float Throttle = 0.0f;

  UPROPERTY(Category = "Vehicle Control", EditAnywhere, BlueprintReadWrite)
  float Steer = 0.0f;

  UPROPERTY(Category = "Vehicle Control", EditAnywhere, BlueprintReadWrite)
  float Brake = 0.0f;

  UPROPERTY(Category = "Vehicle Control", EditAnywhere, BlueprintReadWrite)
  bool bHandBrake = false;

  UPROPERTY(Category = "Vehicle Control", EditAnywhere, BlueprintReadWrite)
  bool bReverse = false;

  UPROPERTY(Category = "Vehicle Control", EditAnywhere, BlueprintReadWrite)
  bool bManualGearShift = false;

  UPROPERTY(Category = "Vehicle Control", EditAnywhere, BlueprintReadWrite)
  float ThrottleFL = 0.0f;
  UPROPERTY(Category = "Vehicle Control", EditAnywhere, BlueprintReadWrite)
  float ThrottleFR = 0.0f;
  UPROPERTY(Category = "Vehicle Control", EditAnywhere, BlueprintReadWrite)
  float ThrottleBL = 0.0f;
  UPROPERTY(Category = "Vehicle Control", EditAnywhere, BlueprintReadWrite)
  float ThrottleBR = 0.0f;

  UPROPERTY(Category = "Vehicle Control", EditAnywhere, BlueprintReadWrite)
  float BrakeFL = 0.0f;
  UPROPERTY(Category = "Vehicle Control", EditAnywhere, BlueprintReadWrite)
  float BrakeFR = 0.0f;
  UPROPERTY(Category = "Vehicle Control", EditAnywhere, BlueprintReadWrite)
  float BrakeBL = 0.0f;
  UPROPERTY(Category = "Vehicle Control", EditAnywhere, BlueprintReadWrite)
  float BrakeBR = 0.0f;


  UPROPERTY(Category = "Vehicle Control", EditAnywhere, BlueprintReadWrite, meta = (EditCondition = bManualGearShift))
  int32 Gear = 0;
};
