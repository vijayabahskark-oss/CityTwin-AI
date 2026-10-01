import carla
import os
import time


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # Get the current world
    world = client.get_world()

    print("Connected to:", world.get_map().name)

    # Get a spawn point
    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No vehicle spawn points available.")

    spawn_point = spawn_points[0]

    # Find a vehicle blueprint
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.*")[0]

    # Spawn vehicle
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    if vehicle is None:
        raise RuntimeError("Vehicle could not be spawned.")

    print("Vehicle spawned:", vehicle.type_id)

    camera = None

    try:
        # RGB camera blueprint
        camera_bp = blueprint_library.find("sensor.camera.rgb")

        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "90")

        # Camera position relative to vehicle
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4)
        )

        camera = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=vehicle
        )

        print("RGB camera spawned.")

        output_dir = "outputs/carla_test"
        os.makedirs(output_dir, exist_ok=True)

        image_saved = False

        def save_image(image):
            nonlocal image_saved

            if not image_saved:
                path = os.path.join(
                    output_dir,
                    "carla_camera.png"
                )

                image.save_to_disk(path)

                print("Image saved:", path)

                image_saved = True

        camera.listen(save_image)

        print("Waiting for camera frame...")

        # Wait for a frame
        for _ in range(50):
            if image_saved:
                break

            time.sleep(0.1)

        if not image_saved:
            print("WARNING: No camera image received.")

    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()

        vehicle.destroy()

        print("Actors cleaned up.")


if __name__ == "__main__":
    main()