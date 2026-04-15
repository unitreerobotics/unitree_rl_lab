#!/usr/bin/env python3
"""
Convert BVH motion data from 29dof format to 23dof format.

29dof joint order:
  0-5: left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
  6-11: right leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
  12-14: waist (yaw, roll, pitch)               <- 23dof: only keeps yaw (idx 12)
  15-21: left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
                                                 <- 23dof: removes wrist_pitch(20), wrist_yaw(21)
  22-28: right arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
                                                 <- 23dof: removes wrist_pitch(26), wrist_yaw(27)

23dof joint order:
  0-5: left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
  6-11: right leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
  12: waist (yaw only)
  13-17: left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll)
  18-22: right arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll)

Mapping from 29dof indices to 23dof indices:
  0-12 → 0-12 (legs + waist_yaw, keep identity)
  15-19 → 13-17 (left arm, skip 20-21)
  22-26 → 18-22 (right arm, skip 27-28)
"""

import argparse
import numpy as np
from pathlib import Path


def convert_bvh_29dof_to_23dof(input_csv: str, output_csv: str | None = None):
    """
    Convert BVH motion data from 29dof to 23dof format.
    
    Args:
        input_csv: Path to input 29dof BVH CSV file
        output_csv: Path to output 23dof BVH CSV file (if None, overwrite input)
    """
    input_path = Path(input_csv)
    
    if output_csv is None:
        # Default: replace .csv with _23dof.csv
        output_path = input_path.parent / f"{input_path.stem}_23dof{input_path.suffix}"
    else:
        output_path = Path(output_csv)
    
    # Load 29dof motion data
    print(f"Loading 29dof motion from: {input_path}")
    motion_29dof = np.loadtxt(input_path, delimiter=",")
    
    num_frames = motion_29dof.shape[0]
    print(f"  Frames: {num_frames}")
    print(f"  Columns: {motion_29dof.shape[1]} (expected 36: 7 base + 29 joints)")
    
    if motion_29dof.shape[1] != 36:
        raise ValueError(f"Expected 36 columns for 29dof motion, got {motion_29dof.shape[1]}")
    
    # Mapping indices: extract base (0-6) and selected joints
    # Base: root position (0-2) + root rotation (3-6)
    # 29dof joints: 7-35
    # 23dof joints selection:
    #   - Keep: 0-12, 15-19, 22-26 (from column indices)
    #   - Skip: 13-14 (waist roll/pitch), 20-21 (left wrist), 27-28 (right wrist)
    
    # Column indices in the CSV file
    # Columns 0-6: base (pos + rot)
    # Columns 7-35: joints (29 joints)
    
    # Joint indices to keep (offset by 7 for column numbers)
    joint_indices_23dof = [
        # Legs + waist yaw (0-12)
        *range(0, 13),
        # Left arm without wrist pitch/yaw (15-19)
        *range(15, 20),
        # Right arm without wrist pitch/yaw (22-26)
        *range(22, 27),
    ]
    
    # Create output array: 7 base + 23 joints = 30 columns
    motion_23dof = np.zeros((num_frames, 7 + 23))
    
    # Copy base (root position + rotation)
    motion_23dof[:, :7] = motion_29dof[:, :7]
    
    # Copy selected joints
    for i, joint_idx in enumerate(joint_indices_23dof):
        col_idx = 7 + joint_idx  # Column index in input
        motion_23dof[:, 7 + i] = motion_29dof[:, col_idx]
    
    # Save output
    print(f"Saving 23dof motion to: {output_path}")
    print(f"  Output shape: {motion_23dof.shape}")
    np.savetxt(output_path, motion_23dof, delimiter=",", fmt="%.6f")
    print("✓ Conversion complete!")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert BVH motion data from 29dof to 23dof format"
    )
    parser.add_argument("input_csv", help="Path to input 29dof BVH CSV file")
    parser.add_argument(
        "-o", "--output",
        help="Path to output 23dof BVH CSV file (default: input_23dof.csv)"
    )
    
    args = parser.parse_args()
    convert_bvh_29dof_to_23dof(args.input_csv, args.output)
