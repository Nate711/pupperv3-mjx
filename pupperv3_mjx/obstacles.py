import xml.etree.ElementTree as ET
import random
from typing import Tuple
import math


def random_z_rotation_quaternion(seed: int = 0):
    """Generates a random quaternion with a random yaw angle."""
    # Generate a random yaw angle
    yaw = random.uniform(-math.pi, math.pi)

    # Construct the quaternion from the yaw angle
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def add_boxes_to_model(
    tree,
    n_boxes: int,
    x_range: Tuple,
    y_range: Tuple,
    height: float = 0.02,
    depth: float = 0.02,
    length: float = 3.0,
    group: str = "0",
    seed: int = 0,
):
    root = tree.getroot()

    # Find the worldbody element
    worldbody = root.find("worldbody")

    # Seed for reproducibility
    random.seed(seed)

    # Add N boxes to the worldbody
    for i in range(n_boxes):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        quat = random_z_rotation_quaternion(seed=seed)
        quat_str = " ".join(map(str, quat))
        ET.SubElement(
            worldbody,
            "geom",
            name=f"box_geom_{i}",
            pos=f"{x} {y} 0",
            quat=quat_str,
            type="box",
            size=f"{depth/2.0} {length/2.0} {height}",
            rgba="0.1 0.5 0.8 1",
            conaffinity="1",
            contype="1",
            condim="3",
            group=group,
        )

    # Convert the modified XML tree back to a string
    return tree

def add_stairs_to_model(
    tree: ET.ElementTree,
    num_stairs: int,
    stair_width: float = 0.1524,
    stair_height: float = 0.04,
    x_start: float = 0.0,
    y_pos: float = 0.0,
    stair_depth: float = 5.0,
    rgba: Tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0),
):
    """
    Add stairs to the MuJoCo model dynamically.

    Args:
        tree (ET.ElementTree): The MuJoCo XML tree to modify.
        num_stairs (int): Number of stairs to add.
        stair_width (float): The width of each stair (x-direction).
        stair_height (float): The height of each stair (z-direction).
        x_start (float): Starting x-position for the stairs.
        y_pos (float): Fixed y-position for all stairs.
        stair_depth (float): Depth of each stair (y-direction).
        rgba (Tuple[float, float, float, float]): Color of the stairs.
    """
    worldbody = tree.find("worldbody")
    for i in range(num_stairs):
        stair_x_pos = x_start + i * stair_width
        stair_z_pos = (i + 1) * stair_height

        # Add a body for each stair
        ET.SubElement(
            worldbody,
            "geom",
            name=f"stair_{i+1}", 
            pos=f"{stair_x_pos} {y_pos} 0",
            type="box",
            size=f"{stair_width / 2} {stair_depth / 2} {stair_z_pos / 2}",
            rgba=" ".join(map(str, rgba)),
            conaffinity="1",
            contype="1",
            condim="3",
            group="0"
        )
    return tree

def add_blocks_to_model(
    tree: ET.ElementTree,
    num_blocks: int,
    block_width: float = 0.02,
    x_start: float = 0.0,
    y_start: float = 0.0,
    block_distance: float = 0.07,
    rgba: Tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0),
):
    """
    Add stairs to the MuJoCo model dynamically.

    Args:
        tree (ET.ElementTree): The MuJoCo XML tree to modify.
        num_stairs (int): Number of stairs to add.
        stair_width (float): The width of each stair (x-direction).
        stair_height (float): The height of each stair (z-direction).
        x_start (float): Starting x-position for the stairs.
        y_pos (float): Fixed y-position for all stairs.
        stair_depth (float): Depth of each stair (y-direction).
        rgba (Tuple[float, float, float, float]): Color of the stairs.
    """
    worldbody = tree.find("worldbody")
    for i in range(num_blocks):
        for j in range(num_blocks):
            block_x_pos = x_start + i * block_distance
            block_y_pos = y_start + j * block_distance
            # Add a body for each stair
            ET.SubElement(
                worldbody,
                "geom",
                name=f"block_{i+1}", 
                pos=f"{block_x_pos} {block_y_pos} 0",
                type="box",
                size=f"{block_width / 2} {block_width / 2} {random.uniform(0.015, 0.04) / 2}",
                rgba=" ".join(map(str, rgba)),
                conaffinity="1",
                contype="1",
                condim="3",
                group="0"
            )
    return tree