import sys
import time
import carla

sys.path.insert(0, ".")

from carla_integration.action_controller import CarlaActionController


def main():

    print("=" * 70)
    print("CITYTWIN AI — CARLA ACTION CONTROLLER TEST")
    print("=" * 70)

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    print(f"Connected to: {world.get_map().name}")

    controller = CarlaActionController(world)

    print("\nInitial traffic-light status:")
    print(controller.get_light_status())

    # Test every DQN action once.
    for action in range(6):

        print("\n" + "#" * 70)
        print(f"TESTING DQN ACTION {action}")
        print("#" * 70)

        result = controller.apply_action(action)

        print("\nResult:")
        print(result)

        print("\nCurrent traffic-light status:")
        print(controller.get_light_status())

        # Give CARLA a moment to process the intervention.
        time.sleep(2)

    print("\n" + "=" * 70)
    print("ACTION CONTROLLER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()