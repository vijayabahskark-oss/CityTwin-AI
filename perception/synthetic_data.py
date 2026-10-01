"""
perception/synthetic_data.py — Synthetic Urban Image Generator
==============================================================
Procedural generator for 128x128 RGB urban traffic camera scenes with
ground-truth multi-label annotations:
  - Label 0: VEHICLE_PRESENT (colored cars/buses on road lanes)
  - Label 1: PEDESTRIAN_PRESENT (pedestrian figures on sidewalks/crosswalks)
  - Label 2: ROAD_OBSTRUCTION (barricades/construction cones on roadway)
  - Label 3: LOW_VISIBILITY (dense fog overlay / rain streaks / contrast drop)

NOTE: This generator produces stylized procedural scenes designed for Unit II
CNN proof-of-concept. It is NOT real-world camera imagery or autonomous driving data.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


LABEL_NAMES: List[str] = [
    "vehicle_present",
    "pedestrian_present",
    "road_obstruction",
    "low_visibility",
]

NUM_LABELS: int = len(LABEL_NAMES)  # 4
IMAGE_SIZE: Tuple[int, int] = (128, 128)


def generate_single_scene(
    has_vehicle: bool,
    has_pedestrian: bool,
    has_obstruction: bool,
    has_low_visibility: bool,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Render a single 128x128 RGB urban scene based on multi-label flags.

    Returns
    -------
    np.ndarray of shape (128, 128, 3), dtype uint8 (RGB)
    """
    r = rng or np.random.RandomState()

    # 1. Base Canvas (128x128)
    img = Image.new("RGB", IMAGE_SIZE, (180, 205, 230))  # Sky
    draw = ImageDraw.Draw(img)

    # 2. Distant Buildings
    skyline_y = r.randint(25, 38)
    num_buildings = r.randint(3, 7)
    bx = 0
    while bx < 128:
        bw = r.randint(15, 35)
        bh = r.randint(15, 30)
        b_color = (r.randint(90, 140), r.randint(95, 145), r.randint(100, 150))
        draw.rectangle([bx, skyline_y - bh, bx + bw, 55], fill=b_color)
        bx += bw + r.randint(2, 6)

    # 3. Sidewalk & Horizon
    draw.rectangle([0, 48, 128, 58], fill=(160, 160, 155))  # Sidewalk
    draw.line([0, 58, 128, 58], fill=(120, 120, 115), width=1)  # Curb

    # 4. Roadway (58 to 128)
    road_gray = r.randint(40, 55)
    draw.rectangle([0, 58, 128, 128], fill=(road_gray, road_gray, road_gray + 5))

    # Centerline markings
    for y in range(65, 128, 14):
        draw.rectangle([62, y, 65, y + 8], fill=(230, 220, 80))

    # 5. Render Pedestrians (if requested)
    if has_pedestrian:
        num_peds = r.randint(1, 4)
        for _ in range(num_peds):
            px = r.randint(8, 115)
            py = r.randint(46, 54)
            shirt_color = (r.randint(180, 240), r.randint(40, 120), r.randint(40, 120))
            # Head
            draw.ellipse([px, py - 6, px + 4, py - 2], fill=(230, 200, 175))
            # Body
            draw.rectangle([px, py - 2, px + 4, py + 4], fill=shirt_color)
            # Legs
            draw.line([px + 1, py + 4, px + 1, py + 8], fill=(40, 40, 60), width=1)
            draw.line([px + 3, py + 4, px + 3, py + 8], fill=(40, 40, 60), width=1)

    # 6. Render Road Obstruction (if requested)
    if has_obstruction:
        num_obs = r.randint(1, 2)
        for _ in range(num_obs):
            ox = r.randint(20, 90)
            oy = r.randint(68, 110)
            ow = r.randint(18, 28)
            oh = r.randint(8, 14)
            # Barricade base
            draw.rectangle([ox, oy, ox + ow, oy + oh], fill=(240, 100, 20))
            # Diagonal warning stripes
            for s in range(ox, ox + ow, 6):
                draw.line([s, oy, s + 3, oy + oh], fill=(255, 255, 255), width=2)

    # 7. Render Vehicles (if requested)
    if has_vehicle:
        num_veh = r.randint(1, 4)
        # Sort by y so lower vehicles render in front
        lanes = [r.randint(62, 78), r.randint(82, 108)]
        for i in range(num_veh):
            vy = lanes[i % len(lanes)] + r.randint(-3, 3)
            # Perspective scaling: vehicles closer to bottom are slightly larger
            scale = 1.0 + (vy - 60) / 70.0
            vw = int(r.randint(20, 28) * scale)
            vh = int(r.randint(10, 14) * scale)
            vx = r.randint(8, max(9, 120 - vw))

            car_colors = [
                (220, 30, 30),   # Red
                (30, 90, 210),   # Blue
                (230, 230, 230), # White
                (35, 35, 40),    # Black
                (30, 160, 60),   # Green
                (240, 180, 20),  # Yellow
            ]
            car_col = car_colors[r.randint(0, len(car_colors))]

            # Car Body
            draw.rectangle([vx, vy, vx + vw, vy + vh], fill=car_col)
            # Roof / Cabin
            rw = int(vw * 0.6)
            rx = vx + int(vw * 0.2)
            draw.rectangle([rx, vy - int(vh * 0.4), rx + rw, vy], fill=(car_col[0] // 2, car_col[1] // 2, car_col[2] // 2))
            # Windshield
            draw.rectangle([rx + 2, vy - int(vh * 0.35), rx + rw - 2, vy - 1], fill=(160, 200, 230))
            # Wheels
            draw.rectangle([vx + 2, vy + vh, vx + 6, vy + vh + 2], fill=(20, 20, 20))
            draw.rectangle([vx + vw - 6, vy + vh, vx + vw - 2, vy + vh + 2], fill=(20, 20, 20))

    img_arr = np.array(img, dtype=np.uint8)

    # 8. Apply Low Visibility / Weather Overlays (if requested)
    if has_low_visibility:
        fog_layer = np.full_like(img_arr, r.randint(190, 220), dtype=np.uint8)
        alpha = r.uniform(0.45, 0.70)
        img_arr = (img_arr.astype(np.float32) * (1.0 - alpha) + fog_layer.astype(np.float32) * alpha).astype(np.uint8)

        # Optional rain streaks
        if r.rand() > 0.3:
            num_rain = r.randint(20, 60)
            for _ in range(num_rain):
                rx1 = r.randint(0, 120)
                ry1 = r.randint(0, 115)
                rlen = r.randint(5, 12)
                img_arr[ry1 : min(128, ry1 + rlen), rx1 : min(128, rx1 + 2)] = (200, 215, 235)

    # 9. Mild sensor noise
    noise = r.normal(0, 3.0, img_arr.shape).astype(np.float32)
    img_arr = np.clip(img_arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img_arr


def generate_synthetic_dataset(
    num_samples: int = 1200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a full balanced synthetic multi-label urban dataset.

    Parameters
    ----------
    num_samples : int
        Total number of images to generate (default 1200).
    seed : int
        Deterministic random seed.

    Returns
    -------
    images : np.ndarray of shape (num_samples, 128, 128, 3), uint8
    labels : np.ndarray of shape (num_samples, 4), float32 in {0.0, 1.0}
    """
    rng = np.random.RandomState(seed)

    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for i in range(num_samples):
        # Sample independent binary flags (each ~50% frequency with combinations)
        has_v = bool(rng.rand() > 0.40)
        has_p = bool(rng.rand() > 0.50)
        has_o = bool(rng.rand() > 0.65)
        has_l = bool(rng.rand() > 0.60)

        # Ensure at least some empty and fully loaded scenes
        if i % 16 == 0:
            has_v, has_p, has_o, has_l = False, False, False, False
        elif i % 16 == 15:
            has_v, has_p, has_o, has_l = True, True, True, True

        img = generate_single_scene(has_v, has_p, has_o, has_l, rng=rng)
        label = np.array([float(has_v), float(has_p), float(has_o), float(has_l)], dtype=np.float32)

        images.append(img)
        labels.append(label)

    return np.stack(images, axis=0), np.stack(labels, axis=0)
