import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import os
from tqdm import tqdm
from math import radians, degrees

# Set random seed for reproducibility
random.seed(42)

# --- Hyperparameters ---
show_live_plot = True  # Set to True to enable live matplotlib plotting of each frame
num_objects = 5         # Number of obstacles per environment
noise_amount = 0.01     # Standard deviation of LIDAR noise
num_simulations = 1    # Number of full simulation runs

# --- Simulated world generation ---
def generate_environment(num_circles=5, area_size=6):
    """Generates a list of circular obstacles randomly placed within a defined area."""
    obstacles = []
    for _ in range(num_circles):
        x, y = random.uniform(-area_size, area_size), random.uniform(-area_size, area_size)
        r = random.uniform(0.5, 1.5)
        obstacles.append({'type': 'circle', 'center': (x, y), 'radius': r})
    return obstacles

# --- LIDAR scan (returns list of (angle, distance)) ---
def lidar_scan(obstacles, n_points=360, max_range=10.0, noise_std=0.05):
    """Simulates a 360-degree LIDAR scan returning noisy and true ranges."""
    clean_ranges = []
    origin = np.array([0.0, 0.0])
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    ranges = []

    for theta in angles:
        dir_vec = np.array([np.cos(theta), np.sin(theta)])
        min_dist = max_range

        for obj in obstacles:
            if obj['type'] == 'circle':
                cx, cy = obj['center']
                r = obj['radius']
                oc = origin - np.array([cx, cy])
                a = 1
                b = 2 * np.dot(oc, dir_vec)
                c = np.dot(oc, oc) - r*r
                disc = b*b - 4*a*c
                if disc >= 0:
                    t = (-b - np.sqrt(disc)) / (2*a)
                    if 0 < t < min_dist:
                        min_dist = t

        noisy_dist = min_dist + np.random.normal(0, noise_std)  # Add noise
        noisy_dist = max(0.0, min(noisy_dist, max_range))        # Clamp distance
        clean_ranges.append((theta, min_dist))
        ranges.append((theta, noisy_dist))
    return ranges, clean_ranges

# --- Find closest object ---
def find_closest(scan):
    """Returns the angle-distance pair closest to the LIDAR origin."""
    return min(scan, key=lambda x: x[1])

# --- Rotate environment ---
def rotate_environment(obstacles, angle_rad):
    """Rotates all obstacles around the origin by a specified angle in radians."""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotated = []
    for obj in obstacles:
        if obj['type'] == 'circle':
            x, y = obj['center']
            x_new = x * cos_a + y * sin_a
            y_new = -x * sin_a + y * cos_a
            rotated.append({'type': 'circle', 'center': (x_new, y_new), 'radius': obj['radius']})
    return rotated

# --- Main loop for generating and saving frames ---
save_animations = False  # Set to True to save .gif animations

import csv
# Create output directory with hyperparameter information
output_dir = f"frames/nearest/objs{num_objects}_noise{noise_amount}_runs{num_simulations}"
os.makedirs(output_dir, exist_ok=True)

# Create CSV file to log summary of each scan
csv_path = os.path.join(output_dir, f"scan_summary.csv")
with open(csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Scan Number", "Rotation at Completion (deg)", "Noisy Closest Distance (m)", "True Closest Distance (m)"])

# Main simulation loop
for run in tqdm(range(num_simulations), desc="Simulations", unit="run"):
    env = generate_environment(num_circles=num_objects, area_size=6)
    reference_scan, _ = lidar_scan(env)
    ref_angle, _ = find_closest(reference_scan)

    # Set up figure and subplots
    fig, (ax_env, ax_plot) = plt.subplots(1, 2, figsize=(14, 6))
    ax_env.set_xlim(-10, 10)
    ax_env.set_ylim(-10, 10)
    ax_env.set_aspect('equal')
    ax_env.set_title("LIDAR Scan at 0° Rotation")

    robot_dot, = ax_env.plot(0, 0, 'ro')  # Robot center
    scan_lines = [ax_env.plot([], [], 'b-', alpha=0.1)[0] for _ in range(360)]  # Laser beams
    closest_dot, = ax_env.plot([], [], 'go')  # Closest point

    obstacle_patches = []
    for obj in env:
        circle = plt.Circle(obj['center'], obj['radius'], fill=False, edgecolor='gray')
        ax_env.add_patch(circle)
        obstacle_patches.append(circle)

    # Plot for angle difference over time
    angle_diffs = []
    x_data = []
    sim_line, = ax_plot.plot([], [], 'm-', label="Angle Difference")
    ax_plot.set_xlim(0, 180)
    ax_plot.set_ylim(0, 180)
    ax_plot.set_xlabel("Rotation (degrees)")
    ax_plot.set_ylabel("Angle Difference (degrees)")
    ax_plot.set_title("Angle Difference to Closest Object")
    ax_plot.axhline(90, color='r', linestyle='--', label='Target 90°')
    ax_plot.legend()

    rotation_angles = np.arange(0, 181, 1)  # 1-degree increments
    frames = []

    # Rotate and scan the environment
    for frame_num, angle_deg in enumerate(rotation_angles):
        angle_rad = radians(angle_deg)
        rotated_env = rotate_environment(env, angle_rad)
        scan, clean_scan = lidar_scan(rotated_env, noise_std=noise_amount)
        closest_theta, closest_dist = find_closest(scan)
        closest_point = np.array([np.cos(closest_theta), np.sin(closest_theta)]) * closest_dist

        # Update LIDAR beams
        for i, (theta, dist) in enumerate(scan):
            x = [0, np.cos(theta) * dist]
            y = [0, np.sin(theta) * dist]
            scan_lines[i].set_data(x, y)

        # Mark closest point
        closest_dot.set_data([closest_point[0]], [closest_point[1]])

        # Rotate obstacles visually
        for patch, obj in zip(obstacle_patches, rotated_env):
            patch.center = obj['center']

        # Update plot showing angle difference
        angle_diff = (closest_theta - ref_angle + np.pi) % (2 * np.pi) - np.pi
        diff_deg = abs(degrees(angle_diff))
        x_data.append(angle_deg)
        angle_diffs.append(diff_deg)
        sim_line.set_data(x_data, angle_diffs)

        # Update title with distances
        true_dist = dict(clean_scan)[closest_theta]
        ax_env.set_title(f"LIDAR Scan at {angle_deg}° Rotation Closest Distance: {closest_dist:.2f} m (true: {true_dist:.2f} m)")

        # Capture frame for animation if enabled
        fig.canvas.draw()
        if show_live_plot:
            plt.pause(0.001)
        if save_animations:
            frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            frame_image = frame_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame_image)

        # Stop rotating once angle difference reaches or exceeds 90°
        if diff_deg >= 90:
            break

    # Save animation if enabled
    if save_animations:
        from PIL import Image
        img_frames = [Image.fromarray(frame) for frame in frames]
        img_frames[0].save(os.path.join(output_dir, f"full_animation_run_{run+1}.gif"), save_all=True, append_images=img_frames[1:], duration=100, loop=0)

    # Save the final frame as a PNG
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"final_frame_run_{run+1}.png"))
    plt.close()

    # Append scan summary to CSV
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([run + 1, angle_deg, f"{closest_dist:.3f}", f"{true_dist:.3f}"])

print(f"Saved final frames, full animations, and scan summary for {num_simulations} runs in 'frames' directory.")
