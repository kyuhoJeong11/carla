// Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#include "SignComponent.h"
#include "/home/junbeom/UnrealEngine_4.24/Engine/Source/Runtime/Engine/Classes/Components/StaticMeshComponent.h"

USignComponent::USignComponent()
{
  PrimaryComponentTick.bCanEverTick = false;
 
 /*
  cubeMesh = ConstructorHelpers::FObjectFinder<UStaticMesh>(TEXT("StaticMesh'/Engine/BasicShapes/Cube.cube'")).Object;
 
  cubeMeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Construct Cube Mesh"));
  cubeMeshComponent->SetStaticMesh(cubeMesh);
  */
}

// Called when the game starts
void USignComponent::BeginPlay()
{
  Super::BeginPlay();

}

// Called every frame
void USignComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

}

const FString& USignComponent::GetSignId() const
{
  return SignId;
}

void USignComponent::SetSignId(const FString &Id) {
  SignId = Id;
}

void USignComponent::InitializeSign(const cr::Map &Map)
{

}

TArray<std::pair<cr::RoadId, const cre::RoadInfoSignal*>>
    USignComponent::GetAllReferencesToThisSignal(const cr::Map &Map)
{
  TArray<std::pair<cr::RoadId, const cre::RoadInfoSignal*>> Result;
  auto waypoints = Map.GenerateWaypointsOnRoadEntries();
  std::unordered_set<cr::RoadId> ExploredRoads;
  for (auto & waypoint : waypoints)
  {
    // Check if we already explored this road
    if (ExploredRoads.count(waypoint.road_id) > 0)
    {
      continue;
    }
    ExploredRoads.insert(waypoint.road_id);

    // Multiple times for same road (performance impact, not in behavior)
    auto SignalReferences = Map.GetLane(waypoint).
        GetRoad()->GetInfos<cre::RoadInfoSignal>();
    for (auto *SignalReference : SignalReferences)
    {
      FString SignalId(SignalReference->GetSignalId().c_str());
      if(SignalId == GetSignId())
      {
        Result.Add({waypoint.road_id, SignalReference});
      }
    }
  }
  return Result;
}

UBoxComponent* USignComponent::GenerateTriggerBox(const FTransform &BoxTransform,
    float BoxSize)
{
  cr::Lane lane;

  int lane_cnt = (lane.GetCnt() - 1) * -1;

  float TotalLaneWidth = 350 * lane_cnt;

  UE_LOG(LogTemp, Warning, TEXT("Total Lane Width : %f"), TotalLaneWidth);

  AActor *ParentActor = GetOwner();
  UBoxComponent *BoxComponent = NewObject<UBoxComponent>(ParentActor);
  //FRotator rot = FRotator(BoxTransform.GetRotation());
  //rot.Yaw += 180;
  BoxComponent->RegisterComponent();
  BoxComponent->AttachToComponent(
      ParentActor->GetRootComponent(),
      FAttachmentTransformRules::KeepRelativeTransform);
  BoxComponent->SetWorldTransform(BoxTransform);
  //BoxComponent->SetWorldRotation(rot);
  BoxComponent->SetBoxExtent(FVector(BoxSize, BoxSize, BoxSize), true);

  FRotator rot = FRotator(ParentActor->GetRootComponent()->GetOwner()->GetTransform().GetRotation());

  UE_LOG(LogTemp, Warning, TEXT("Rot : %f"), rot.Yaw);

  int tmp;
  FVector vec = BoxTransform.GetLocation();

  if(90.0 < rot.Yaw && rot.Yaw <= 180.0) {
    tmp = 90 - (rot.Yaw - 90);

    int x = TotalLaneWidth / 90 * (rot.Yaw - 90);
    int y = 400 - (TotalLaneWidth / 90 * tmp);

    vec.X -= x;
    vec.Y += y;
  }
  else if(0.0 < rot.Yaw && rot.Yaw <= 90.0) {
    tmp = 90 - rot.Yaw;

    int x = TotalLaneWidth / 90 * tmp - 400;
    int y = TotalLaneWidth / 90 * rot.Yaw;

    vec.X -= x;
    vec.Y += y;
  }
  else if(-90.0 < rot.Yaw && rot.Yaw <= 0.0) {
    tmp = 90 + rot.Yaw;

    int x = TotalLaneWidth / 90 * tmp;
    int y = TotalLaneWidth / 90 * rot.Yaw + 400;

    vec.X += x;
    vec.Y -= y;
  }
  else if(-180.0 < rot.Yaw && rot.Yaw <= -90.0) {
    tmp = 90 + (rot.Yaw + 90);

    int x = TotalLaneWidth / 90 * (rot.Yaw + 90) + 400;
    int y = TotalLaneWidth / 90 * tmp;

    vec.X -= x;
    vec.Y -= y;
  }

  BoxComponent->SetWorldLocation(vec);

  FString str1 = ParentActor->GetRootComponent()->GetOwner()->GetName();

  FVector vec2 = BoxTransform.GetLocation();

  vec2.X = vec.X;
  vec2.Y = vec.Y;

  FString str2 = BoxTransform.GetLocation().ToString();

  FString str3 = vec2.ToString();
  
  UE_LOG(LogTemp, Warning, TEXT("Name : %s, Loc : %s, After Loc : %s"), *str1, *str2, *str3);
  
  
  //cubeMeshComponent->SetStaticMesh(cubeMesh);
  //cubeMeshComponent->AttachToComponent(BoxComponent, FAttachmentTransformRules::KeepRelativeTransform);
  
  //RootComponent = cubeMeshComponent;

  return BoxComponent;
}

void USignComponent::AddEffectTriggerVolume(UBoxComponent* TriggerVolume)
{
  EffectTriggerVolumes.Add(TriggerVolume);
}

const TArray<UBoxComponent*> USignComponent::GetEffectTriggerVolume() const
{
  return EffectTriggerVolumes;
}
