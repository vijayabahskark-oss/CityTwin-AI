import carla

from carla_integration.state_adapter import CarlaStateAdapter


def main():

    client = carla.Client(
        "localhost",
        2000,
    )

    client.set_timeout(10.0)

    world = client.get_world()

    adapter = CarlaStateAdapter(
        world,
        scenario_tag="carla_test",
    )

    state = adapter.get_state(
        timestep=0,
    )

    print("\nCARLA → CityState")
    print("-----------------")

    print(state)

    print("\nDictionary:")
    print(state.to_dict())


if __name__ == "__main__":
    main()