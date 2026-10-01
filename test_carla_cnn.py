import carla
import os
import time

from perception.predictor import UrbanPerceptionPredictor


CARLA_HOST = "localhost"
CARLA_PORT = 2000
MODEL_PATH = "models/urban_cnn.pt"


def main():
    # -----------------------------
    # Load CNN
    # -----------------------------
    predictor = UrbanPerceptionPredictor.from_checkpoint(MODEL_PATH)

    print("CNN loaded successfully.")

    # -----------------------------
    # Connect to CARLA
    # -----------------------------
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)

    world = client.get_world()

    print("Connected to:", world.get_map().name)

    blueprint_library = world.get_blueprint_library()

    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points available.")

    # -----------------------------
    # Spawn vehicle
    # -----------------------------
    vehicle_bp = blueprint_library.filter("vehicle.*")[0]

    vehicle = world.try_spawn_actor(
        vehicle_bp,
        spawn_points[0]
    )

    if vehicle is None:
        raise RuntimeError("Could not spawn vehicle.")

    camera = None

    try:
        # -----------------------------
        # RGB camera
        # -----------------------------
        camera_bp = blueprint_library.find(
            "sensor.camera.rgb"
        )

        camera_bp.set_attribute(
            "image_size_x",
            "128"
        )

        camera_bp.set_attribute(
            "image_size_y",
            "128"
        )

        camera_bp.set_attribute(
            "fov",
            "90"
        )

        camera_transform = carla.Transform(
            carla.Location(
                x=1.5,
                z=2.4
            )
        )

        camera = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=vehicle
        )

        print("RGB camera spawned.")

        processed = False

        def process_image(image):
            nonlocal processed

            if processed:
                return

            # CARLA image → raw BGRA bytes
            image_data = image.raw_data

            import numpy as np

            array = np.frombuffer(
                image_data,
                dtype=np.uint8
            )

            array = array.reshape(
                (image.height, image.width, 4)
            )

            # BGRA → RGB
            rgb = array[:, :, :3][:, :, ::-1].copy()

            # CNN inference
            prediction = predictor.predict(rgb)

            print("\nCNN prediction:")
            for label, probability in prediction.items():
                print(
                    f"  {label}: "
                    f"{probability:.4f}"
                )

            # Save the first CARLA frame
            output_dir = "outputs/carla_test"

            os.makedirs(
                output_dir,
                exist_ok=True
            )

            image.save_to_disk(
                os.path.join(
                    output_dir,
                    "carla_cnn_frame"
                )
            )

            print(
                "\nCARLA frame processed "
                "successfully."
            )

            processed = True

        camera.listen(process_image)

        print(
            "Waiting for CARLA camera frame..."
        )

        for _ in range(100):

            if processed:
                break

            time.sleep(0.1)

        if not processed:
            print(
                "WARNING: "
                "No frame processed."
            )

    finally:

        if camera is not None:
            camera.stop()
            camera.destroy()

        vehicle.destroy()

        print(
            "\nCARLA actors cleaned up."
        )


if __name__ == "__main__":
    main()