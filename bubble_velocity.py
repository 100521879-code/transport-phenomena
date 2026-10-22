#!/usr/bin/env python3

import argparse
import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def find_default_video():
    """Look for likely video names in the workspace and common subfolders."""
    base_dir = os.getcwd()
    candidate_dirs = [base_dir, os.path.join(base_dir, "Data. Lab1")]
    candidates = [
        "water fluids.MOV",
        "waters fluids.MOV",
        "fluidsglycerol.MOV",
        "water fluids.mov",
        "waters fluids.mov",
        "fluidsglycerol.mov",
    ]

    seen = set()
    for directory in candidate_dirs:
        for name in candidates:
            full_path = os.path.join(directory, name)
            if full_path not in seen and os.path.exists(full_path):
                seen.add(full_path)
                return full_path

    return None


def select_frame_for_calibration(video_path):
    """Open the video, display frames, and let the user select a clear frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Video read failed during calibration.")

        display = frame.copy()
        cv2.putText(
            display,
            "Choose a frame with the reference scale visible. Press n/p to move, Enter to accept.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Video calibration frame", display)
        key = cv2.waitKey(0) & 0xFF

        if key in (13, 10):
            cv2.destroyAllWindows()
            return frame, current_frame
        if key == ord("n"):
            current_frame = min(current_frame + 1, total_frames - 1)
        elif key == ord("p"):
            current_frame = max(current_frame - 1, 0)
        elif key in (27, ord("q")):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt


def get_reference_points(frame):
    """Let the user click the two ends of a known reference length."""
    points = []
    window_name = "Reference length calibration"

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, callback)

    while True:
        display = frame.copy()
        for idx, point in enumerate(points, start=1):
            cv2.circle(display, point, 6, (0, 255, 0), -1)
            cv2.putText(display, str(idx), (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Click the first end of the reference length, then the second end.", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if len(points) == 2:
            cv2.destroyAllWindows()
            return points
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt


def calibrate_pixels_to_cm(video_path):
    """Prompt the user for a known length in the video and return px/cm conversion."""
    frame, _ = select_frame_for_calibration(video_path)
    reference_points = get_reference_points(frame)

    known_length_cm = float(input("Enter the known reference length in centimetres: ").strip())
    dx = reference_points[1][0] - reference_points[0][0]
    dy = reference_points[1][1] - reference_points[0][1]
    reference_pixels = np.hypot(dx, dy)

    if reference_pixels <= 0:
        raise ValueError("Reference points must be distinct.")

    pixels_per_cm = reference_pixels / known_length_cm
    return pixels_per_cm


def track_bubble_manually(video_path, pixels_per_cm, step_frames=20):
    """Allow manual bubble-centre picking every step_frames frames and return time/position data."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 1.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    tracked = []
    click = None
    origin_y = None

    def callback(event, x, y, flags, param):
        nonlocal click
        if event == cv2.EVENT_LBUTTONDOWN:
            click = (x, y)

    cv2.namedWindow("Bubble tracking", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Bubble tracking", callback)

    print(f"Tracking every {step_frames} frames. Press q to finish.")

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ok, frame = cap.read()
        if not ok:
            break

        display = frame.copy()
        cv2.putText(
            display,
            f"Frame {current_frame + 1}/{total_frames}. Click bubble centre every {step_frames} frames. Press q to finish.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if click is not None:
            x_px, y_px = click
            cv2.circle(display, (x_px, y_px), 6, (0, 0, 255), -1)
            if origin_y is None:
                origin_y = y_px
                position_cm = 0.0
            else:
                position_cm = (origin_y - y_px) / pixels_per_cm

            time_s = current_frame / fps
            tracked.append(
                {
                    "frame": current_frame,
                    "time_s": time_s,
                    "x_px": float(x_px),
                    "y_px": float(y_px),
                    "position_cm": float(position_cm),
                }
            )
            print(f"Saved frame {current_frame}: time = {time_s:.3f} s, bubble centre = ({x_px}, {y_px}), position = {position_cm:.3f} cm")
            click = None
            current_frame += step_frames
            if current_frame >= total_frames:
                break

        cv2.imshow("Bubble tracking", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("n"):
            current_frame = min(current_frame + step_frames, total_frames - 1)
        elif key == ord("p"):
            current_frame = max(current_frame - step_frames, 0)
        elif key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not tracked:
        raise RuntimeError("No bubble positions were selected. Nothing to analyse.")

    return tracked


def estimate_terminal_velocity(time_s, position_cm):
    """Estimate the approximately constant-speed terminal velocity from the later part of the motion."""
    velocity = np.gradient(position_cm, time_s)
    if len(velocity) < 3:
        return float(np.median(velocity)), velocity

    window = max(3, len(velocity) // 4)
    best_window = None
    best_score = float("inf")

    for start in range(0, len(velocity) - window + 1):
        segment = velocity[start : start + window]
        mean_v = np.mean(segment)
        std_v = np.std(segment)
        score = std_v / (abs(mean_v) + 1e-12)
        if score < best_score:
            best_score = score
            best_window = segment
            best_start = start

    if best_window is None:
        best_start = len(velocity) - window
        best_window = velocity[best_start:]

    terminal_velocity = float(np.median(best_window))
    return terminal_velocity, velocity


def save_data_csv(filename, tracked_data):
    """Save tracked data to CSV for the lab report."""
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "time_s", "x_px", "y_px", "position_cm"])
        for row in tracked_data:
            writer.writerow([row["frame"], row["time_s"], row["x_px"], row["y_px"], row["position_cm"]])


def plot_results(time_s, position_cm, velocity_cm_s, terminal_velocity):
    """Create and save the position and velocity plots."""
    pos_fig, pos_ax = plt.subplots(figsize=(8, 5))
    pos_ax.plot(time_s, position_cm, "o-", color="tab:blue", linewidth=2)
    pos_ax.set_xlabel("Time (s)")
    pos_ax.set_ylabel("Position (cm)")
    pos_ax.set_title("Bubble position versus time")
    pos_ax.grid(True, alpha=0.3)
    pos_fig.tight_layout()
    pos_fig.savefig("bubble_position_vs_time.png", dpi=300)

    vel_fig, vel_ax = plt.subplots(figsize=(8, 5))
    vel_ax.plot(time_s, velocity_cm_s, "o-", color="tab:orange", linewidth=2)
    vel_ax.axhline(terminal_velocity, color="black", linestyle="--", linewidth=1.5, label=f"Estimated terminal velocity = {terminal_velocity:.3f} cm/s")
    vel_ax.set_xlabel("Time (s)")
    vel_ax.set_ylabel("Velocity (cm/s)")
    vel_ax.set_title("Bubble velocity versus time")
    vel_ax.grid(True, alpha=0.3)
    vel_ax.legend()
    vel_fig.tight_layout()
    vel_fig.savefig("bubble_velocity_vs_time.png", dpi=300)

    plt.close("all")


def main():
    parser = argparse.ArgumentParser(
        description="Manually track a rising bubble in a video, convert pixels to centimetres, and plot time-position and time-velocity data."
    )
    parser.add_argument("video", nargs="?", help="Path to the MOV video file. Defaults to a matching file in the working directory.")
    parser.add_argument("--reference-length", type=float, default=None, help="Known reference length in centimetres for calibration.")
    args = parser.parse_args()

    video_path = args.video or find_default_video()
    if not video_path:
        raise FileNotFoundError(
            "No video file found in the current folder or Data. Lab1. "
            "Use a real filename such as python bubble_velocity.py \"water fluids.MOV\" or "
            "python bubble_velocity.py \"Data. Lab1\\fluidsglycerol.MOV\""
        )

    print(f"Using video: {video_path}")

    if args.reference_length is None:
        print("You will now pick a known reference length in the video.")
        pixels_per_cm = calibrate_pixels_to_cm(video_path)
    else:
        frame, _ = select_frame_for_calibration(video_path)
        reference_points = get_reference_points(frame)
        dx = reference_points[1][0] - reference_points[0][0]
        dy = reference_points[1][1] - reference_points[0][1]
        reference_pixels = np.hypot(dx, dy)
        pixels_per_cm = reference_pixels / args.reference_length

    print(f"Calibration: {pixels_per_cm:.3f} px per cm")
    tracked = track_bubble_manually(video_path, pixels_per_cm)

    time_s = np.array([row["time_s"] for row in tracked], dtype=float)
    position_cm = np.array([row["position_cm"] for row in tracked], dtype=float)
    velocity_cm_s = np.gradient(position_cm, time_s)

    terminal_velocity, _ = estimate_terminal_velocity(time_s, position_cm)
    print(f"Estimated terminal velocity: {terminal_velocity:.3f} cm/s")

    save_data_csv("bubble_tracking_data.csv", tracked)
    plot_results(time_s, position_cm, velocity_cm_s, terminal_velocity)

    print("Saved data to bubble_tracking_data.csv")
    print("Saved plots to bubble_position_vs_time.png and bubble_velocity_vs_time.png")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTracking cancelled by user.")
