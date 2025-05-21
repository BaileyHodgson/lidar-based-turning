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
show_live_plot = True  # Set to True to enable live plotting in matplotlib
stop_similarity_threshold = -2  # Stop rotation when similarity exceeds this value
num_objects = 8         # Number of obstacles per environment
noise_amount = 0.05     # Standard deviation of LIDAR noise
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

# --- Compute similarity ---
def scan_similarity(scan1, scan2):
    """Returns similarity (inverse error) between two LIDAR scans."""
    distances1 = np.array([d for _, d in scan1])
    distances2 = np.array([d for _, d in scan2])
    return -np.linalg.norm(distances1 - distances2)  # Higher is more similar

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
output_dir = f"frames/objs{num_objects}_noise{noise_amount}_runs{num_simulations}"
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, f"scan_summary.csv")
with open(csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Scan Number", "Rotation at Completion (deg)", "Best Similarity"])

for run in tqdm(range(num_simulations), desc="Simulations", unit="run"):
    env = generate_environment(num_circles=num_objects, area_size=6)
    initial_scan, _ = lidar_scan(env, noise_std=noise_amount)

    # Rotate environment by 90° and take a new reference scan
    rotated_env_for_ref = rotate_environment(env, radians(90))
    reference_scan, _ = lidar_scan(rotated_env_for_ref, noise_std=noise_amount)  # 90 degree rotation

    fig, (ax_env, ax_sim) = plt.subplots(1, 2, figsize=(14, 6))
    ax_env.set_xlim(-10, 10)
    ax_env.set_ylim(-10, 10)
    ax_env.set_aspect('equal')
    ax_env.set_title("LIDAR Scan Rotation")

    scan_lines = [ax_env.plot([], [], 'b-', alpha=0.1)[0] for _ in range(360)]
    robot_dot, = ax_env.plot(0, 0, 'ro')
    scan_points, = ax_env.plot([], [], 'b.', label='Current Scan')
    ref_points, = ax_env.plot([], [], 'g.', alpha=0.5, label='Reference Scan')
    
    obstacle_patches = []
    for obj in env:
        circle = plt.Circle(obj['center'], obj['radius'], fill=False, edgecolor='gray')
        ax_env.add_patch(circle)
        obstacle_patches.append(circle)

    rotation_angles = np.arange(0, 181, 1)
    similarity_scores = []
    x_vals = []
    sim_line, = ax_sim.plot([], [], 'm-', label='Similarity')
    ax_sim.set_xlim(0, 180)
    # ax_sim.set_ylim(-200, 0)  # Removed fixed limits to enable autoscaling
    ax_sim.set_title("Scan Similarity vs Rotation")
    ax_sim.set_xlabel("Rotation (degrees)")
    ax_sim.set_ylabel("Similarity (negative L2 distance)")
    ax_sim.legend()
    best_similarity = -np.inf
    best_angle = 0
    frames = []

    for angle_deg in rotation_angles:
        angle_rad = radians(angle_deg)
        rotated_env = rotate_environment(env, angle_rad)
        scan, _ = lidar_scan(rotated_env, noise_std=noise_amount)

        similarity = scan_similarity(scan, reference_scan)
        similarity_scores.append(similarity)
        x_vals.append(angle_deg)
        sim_line.set_data(x_vals, similarity_scores)
        ax_sim.relim()
        ax_sim.autoscale_view()

        # Plot LIDAR scan rays
        for i, (theta, dist) in enumerate(scan):
            x = [0, np.cos(theta) * dist]
            y = [0, np.sin(theta) * dist]
            scan_lines[i].set_data(x, y)

        # Update scan point clouds
        scan_arr = np.array([[np.cos(theta) * dist, np.sin(theta) * dist] for theta, dist in scan])
        ref_arr = np.array([[np.cos(theta) * dist, np.sin(theta) * dist] for theta, dist in reference_scan])
        scan_points.set_data(scan_arr[:, 0], scan_arr[:, 1])
        ref_points.set_data(ref_arr[:, 0], ref_arr[:, 1])

        for patch, obj in zip(obstacle_patches, rotated_env):
            patch.center = obj['center']

        ax_env.set_title(f"Rotation: {angle_deg}°, Similarity: {similarity:.2f}")
        ax_env.legend()
        ax_env.axis('off')  # Hide axis ticks for cleaner output

        if show_live_plot:
            plt.pause(0.001)
        fig.canvas.draw()
        fig.canvas.flush_events()  # Ensure rendering updates
        if save_animations:
            frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            frame_image = frame_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame_image)

        if similarity > best_similarity:
            best_similarity = similarity
            best_angle = angle_deg

        if similarity >= stop_similarity_threshold:
            break

        similarity = scan_similarity(scan, reference_scan)
        
        similarity = scan_similarity(scan, reference_scan)
        similarity_scores.append(similarity)
        x_vals.append(angle_deg)
        sim_line.set_data(x_vals, similarity_scores)
        ax_sim.relim()
        ax_sim.autoscale_view()

        # Plot LIDAR scan rays
        for i, (theta, dist) in enumerate(scan):
            x = [0, np.cos(theta) * dist]
            y = [0, np.sin(theta) * dist]
            scan_lines[i].set_data(x, y)

        # Update scan point clouds
        scan_arr = np.array([[np.cos(theta) * dist, np.sin(theta) * dist] for theta, dist in scan])
        ref_arr = np.array([[np.cos(theta) * dist, np.sin(theta) * dist] for theta, dist in reference_scan])
        scan_points.set_data(scan_arr[:, 0], scan_arr[:, 1])
        ref_points.set_data(ref_arr[:, 0], ref_arr[:, 1])
        
        for patch, obj in zip(obstacle_patches, rotated_env):
            patch.center = obj['center']

        ax_env.set_title(f"Rotation: {angle_deg}°, Similarity: {similarity:.2f}")
        ax_env.legend()
        ax_env.axis('off')  # Hide axis ticks for cleaner output

        fig.canvas.draw()
        fig.canvas.flush_events()  # Ensure rendering updates
        if save_animations:
            frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            frame_image = frame_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame_image)

        if similarity > best_similarity:
            best_similarity = similarity
            best_angle = angle_deg

    if save_animations:
        from PIL import Image
        img_frames = [Image.fromarray(frame) for frame in frames]
        img_frames[0].save(os.path.join(output_dir, f"full_animation_run_{run+1}.gif"), save_all=True, append_images=img_frames[1:], duration=100, loop=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"final_frame_run_{run+1}.png"))
    plt.close()

    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([run + 1, best_angle, f"{best_similarity:.4f}"])

print("Saved final frames, full animations, and scan similarity summary for all runs.")
