import carla


class CarlaActionController:
    """
    Applies the DQN-selected intervention to the CARLA world.

    DQN actions:
        0 - NO_ACTION
        1 - EXTEND_GREEN
        2 - REDUCE_GREEN
        3 - TRAFFIC_DIVERSION
        4 - EMERGENCY_PRIORITY
        5 - COMBINED_RESPONSE
    """

    ACTION_NAMES = {
        0: "NO_ACTION",
        1: "EXTEND_GREEN",
        2: "REDUCE_GREEN",
        3: "TRAFFIC_DIVERSION",
        4: "EMERGENCY_PRIORITY",
        5: "COMBINED_RESPONSE",
    }

    def __init__(self, world):
        self.world = world
        self.traffic_lights = list(
            world.get_actors().filter("traffic.traffic_light")
        )

        print(
            f"ActionController initialized "
            f"with {len(self.traffic_lights)} traffic lights."
        )

    def _get_target_light(self):
        """Return one usable traffic light for the MVP controller."""
        if not self.traffic_lights:
            return None

        return self.traffic_lights[0]

    def _set_light(self, light, state):
        """Safely change the traffic-light state."""
        light.set_state(state)

    def no_action(self):
        """Leave CARLA unchanged."""
        print("NO_ACTION → CARLA unchanged.")

    def extend_green(self, light):
        """
        Extend green duration.

        For the MVP we freeze the selected light in GREEN briefly.
        """
        if light is None:
            return

        light.set_state(carla.TrafficLightState.Green)
        light.set_green_time(15.0)

        print(
            "EXTEND_GREEN → selected traffic light set to GREEN "
            "with extended green duration."
        )

    def reduce_green(self, light):
        """
        Reduce green duration.

        The light is allowed to transition sooner by using
        a short green duration.
        """
        if light is None:
            return

        light.set_green_time(3.0)

        print(
            "REDUCE_GREEN → selected traffic light green duration "
            "reduced."
        )

    def traffic_diversion(self, light):
        """
        MVP traffic diversion.

        We use a red signal at the selected intersection to represent
        restricting the affected approach and diverting traffic.
        """
        if light is None:
            return

        light.set_state(carla.TrafficLightState.Red)
        light.set_red_time(8.0)

        print(
            "TRAFFIC_DIVERSION → selected approach restricted "
            "using RED signal."
        )

    def emergency_priority(self, light):
        """
        Give priority to emergency traffic.

        MVP implementation:
        selected traffic light immediately becomes GREEN.
        """
        if light is None:
            return

        light.set_state(carla.TrafficLightState.Green)
        light.set_green_time(20.0)

        print(
            "EMERGENCY_PRIORITY → selected traffic light "
            "given extended GREEN priority."
        )

    def combined_response(self, light):
        """
        Combined intervention:
        prioritize the selected intersection and extend green.
        """
        if light is None:
            return

        light.set_state(carla.TrafficLightState.Green)
        light.set_green_time(20.0)

        print(
            "COMBINED_RESPONSE → emergency priority + "
            "extended GREEN applied."
        )

    def apply_action(self, action):
        """
        Apply a DQN action to CARLA.

        Returns a dictionary describing the intervention.
        """

        action = int(action)

        if action not in self.ACTION_NAMES:
            raise ValueError(f"Unknown DQN action: {action}")

        action_name = self.ACTION_NAMES[action]

        print("\n" + "=" * 60)
        print("CARLA ACTION")
        print("=" * 60)
        print(f"Action: {action}")
        print(f"Name  : {action_name}")

        light = self._get_target_light()

        if light is None:
            print("WARNING: No traffic lights available.")
            return {
                "action": action,
                "action_name": action_name,
                "applied": False,
            }

        if action == 0:
            self.no_action()

        elif action == 1:
            self.extend_green(light)

        elif action == 2:
            self.reduce_green(light)

        elif action == 3:
            self.traffic_diversion(light)

        elif action == 4:
            self.emergency_priority(light)

        elif action == 5:
            self.combined_response(light)

        return {
            "action": action,
            "action_name": action_name,
            "applied": True,
            "traffic_light_id": light.id,
            "traffic_light_state": str(light.get_state()),
        }

    def get_light_status(self):
        """Return the current state of the controlled traffic light."""

        light = self._get_target_light()

        if light is None:
            return None

        return {
            "id": light.id,
            "state": str(light.get_state()),
            "green_time": light.get_green_time(),
            "yellow_time": light.get_yellow_time(),
            "red_time": light.get_red_time(),
        }