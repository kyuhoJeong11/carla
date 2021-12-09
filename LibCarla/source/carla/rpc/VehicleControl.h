// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/MsgPack.h"

#ifdef LIBCARLA_INCLUDED_FROM_UE4
#  include "Carla/Vehicle/VehicleControl.h"
#endif // LIBCARLA_INCLUDED_FROM_UE4

namespace carla {
namespace rpc {

  class VehicleControl {
  public:

    VehicleControl() = default;

    VehicleControl(
        float in_throttle,
        float in_steer,
        float in_brake,
        bool in_hand_brake,
        bool in_reverse,
        bool in_manual_gear_shift,
        int32_t in_gear)
      : throttle(in_throttle),
        steer(in_steer),
        brake(in_brake),
        hand_brake(in_hand_brake),
        reverse(in_reverse),
        manual_gear_shift(in_manual_gear_shift),
        gear(in_gear),
        throttle_fl(0.0f),
        throttle_fr(0.0f),
        throttle_bl(0.0f),
        throttle_br(0.0f), 
        brake_fl(0.0f),
        brake_fr(0.0f),
        brake_bl(0.0f),
        brake_br(0.0f)  {}

    VehicleControl(
        float in_throttle,
        float in_steer,
        float in_brake,
        bool in_hand_brake,
        bool in_reverse,
        bool in_manual_gear_shift,
        int32_t in_gear,
        float in_throttle_fl,
        float in_throttle_fr,
        float in_throttle_bl,
        float in_throttle_br,
        float in_brake_fl,
        float in_brake_fr,
        float in_brake_bl,
        float in_brake_br)
      : throttle(in_throttle),
        steer(in_steer),
        brake(in_brake),
        hand_brake(in_hand_brake),
        reverse(in_reverse),
        manual_gear_shift(in_manual_gear_shift),
        gear(in_gear),
        throttle_fl(in_throttle_fl),
        throttle_fr(in_throttle_fr),
        throttle_bl(in_throttle_bl),
        throttle_br(in_throttle_br), 
        brake_fl(in_brake_fl),
        brake_fr(in_brake_fr),
        brake_bl(in_brake_bl),
        brake_br(in_brake_br)  {}

    float throttle = 0.0f;
    float steer = 0.0f;
    float brake = 0.0f;
    bool hand_brake = false;
    bool reverse = false;
    bool manual_gear_shift = false;
    int32_t gear = 0;
    float throttle_fl = 0.0f;
    float throttle_fr = 0.0f;
    float throttle_bl = 0.0f;
    float throttle_br = 0.0f;
    float brake_fl = 0.0f;
    float brake_fr = 0.0f;
    float brake_bl = 0.0f;
    float brake_br = 0.0f;


#ifdef LIBCARLA_INCLUDED_FROM_UE4

    VehicleControl(const FVehicleControl &Control)
      : throttle(Control.Throttle),
        steer(Control.Steer),
        brake(Control.Brake),
        hand_brake(Control.bHandBrake),
        reverse(Control.bReverse),
        manual_gear_shift(Control.bManualGearShift),
        gear(Control.Gear),
        throttle_fl(Control.ThrottleFL),
        throttle_fr(Control.ThrottleFR),
        throttle_bl(Control.ThrottleBL),
        throttle_br(Control.ThrottleBR), 
        brake_fl(Control.BrakeFL),
        brake_fr(Control.BrakeFR),
        brake_bl(Control.BrakeBL),
        brake_br(Control.BrakeBR) {}

    operator FVehicleControl() const {
      FVehicleControl Control;
      Control.Throttle = throttle;
      Control.Steer = steer;
      Control.Brake = brake;
      Control.bHandBrake = hand_brake;
      Control.bReverse = reverse;
      Control.bManualGearShift = manual_gear_shift;
      Control.Gear = gear;
      Control.ThrottleFL = throttle_fl;
      Control.ThrottleFR = throttle_fr;
      Control.ThrottleBL = throttle_bl;
      Control.ThrottleBR = throttle_br;
      Control.BrakeFL = brake_fl;
      Control.BrakeFR = brake_fr;
      Control.BrakeBL = brake_bl;
      Control.BrakeBR = brake_br;
      return Control;
    }

#endif // LIBCARLA_INCLUDED_FROM_UE4

    bool operator!=(const VehicleControl &rhs) const {
      return
          throttle != rhs.throttle ||
          steer != rhs.steer ||
          brake != rhs.brake ||
          hand_brake != rhs.hand_brake ||
          reverse != rhs.reverse ||
          manual_gear_shift != rhs.manual_gear_shift ||
          gear != rhs.gear ||
          throttle_fl != rhs.throttle_fl ||
          throttle_fr != rhs.throttle_fr ||
          throttle_bl != rhs.throttle_bl ||
          throttle_br != rhs.throttle_br ||
          brake_fl != rhs.brake_fl ||
          brake_fr != rhs.brake_fr ||
          brake_bl != rhs.brake_bl ||
          brake_br != rhs.brake_br;
    }

    bool operator==(const VehicleControl &rhs) const {
      return !(*this != rhs);
    }

    MSGPACK_DEFINE_ARRAY(
        throttle,
        steer,
        brake,
        hand_brake,
        reverse,
        manual_gear_shift,
        gear,
        throttle_fl,
        throttle_fr,
        throttle_bl,
        throttle_br,
        brake_fl,
        brake_fr,
        brake_bl,
        brake_br);
  };

} // namespace rpc
} // namespace carla
