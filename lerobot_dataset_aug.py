#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset Augmentation Script for Language Instructions

This script augments a robotics dataset by creating multiple copies with different
language instructions for the same task. This helps train vision-language-action models
to be more robust to language variations.

Usage examples:
    # Interactive mode
    lerobot-dataset-aug --dataset.repo_id="roboseasy/Pick_up_a_sock_and_place_it_in_the_basket"
    
    # Using instruction file
    lerobot-dataset-aug \\
        --dataset.repo_id="roboseasy/Pick_up_a_sock_and_place_it_in_the_basket" \\
        --instructions_file="instructions.txt"
    
    # Direct instructions
    lerobot-dataset-aug \\
        --dataset.repo_id="roboseasy/Pick_up_a_sock_and_place_it_in_the_basket" \\
        --instructions "pick the socks and place it into box" \\
        --instructions "grab a sock and put it in the basket"
"""

import argparse
import logging
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets, modify_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_instructions_from_file(file_path: str) -> list[str]:
    """Read instructions from a text file (one per line)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Instructions file not found: {file_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        instructions = [line.strip() for line in f if line.strip()]
    
    if not instructions:
        raise ValueError(f"No instructions found in file: {file_path}")
    
    logger.info(f"Loaded {len(instructions)} instructions from {file_path}")
    return instructions


def get_instructions_interactive() -> list[str]:
    """Get instructions from user via interactive input."""
    print("\n" + "="*60)
    print("증강시킬 instruction을 입력해주세요 (빈 줄 입력 시 종료):")
    print("="*60 + "\n")
    
    instructions = []
    idx = 1
    
    while True:
        try:
            instruction = input(f"[{idx}] ").strip()
            if not instruction:
                break
            instructions.append(instruction)
            idx += 1
        except (KeyboardInterrupt, EOFError):
            print("\n입력이 중단되었습니다.")
            break
    
    if not instructions:
        raise ValueError("최소 1개 이상의 instruction을 입력해야 합니다.")
    
    print(f"\n총 {len(instructions)}개의 instruction이 입력되었습니다.\n")
    return instructions


def save_instructions_to_file(instructions: list[str], file_path: str = "instructions.txt"):
    """Save instructions to a text file."""
    path = Path(file_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(instructions))
    logger.info(f"Instructions saved to {file_path}")


def augment_dataset(
    base_repo_id: str,
    instructions: list[str],
    output_repo_id: str | None = None,
    push_to_hub: bool = True,
):
    """Augment dataset with multiple language instructions."""
    if output_repo_id is None:
        output_repo_id = base_repo_id + "_AUG"
    
    logger.info(f"Loading original dataset: {base_repo_id}")
    original_dataset = LeRobotDataset(base_repo_id)
    
    logger.info(f"Original dataset has {original_dataset.meta.total_episodes} episodes")
    logger.info(f"Creating {len(instructions)} augmented versions...")

    augmented_datasets = []

    for i, instruction in enumerate(instructions):
        logger.info(f"\n[{i+1}/{len(instructions)}] Processing instruction: '{instruction}'")
        
        # Create a function that returns the new task for each frame
        def create_task_fn(task_text):
            def task_fn(row_dict, episode_index, frame_index):
                return task_text
            return task_fn
        
        task_fn = create_task_fn(instruction)
        
        # Create augmented dataset with modified task
        temp_repo_id = f"{base_repo_id}_temp_{i}"
        
        try:
            augmented_ds = modify_features(
                original_dataset,
                add_features={
                    "task": (
                        task_fn,
                        {"dtype": "string", "shape": (1,), "names": None}
                    )
                },
                remove_features=["task"],
                repo_id=temp_repo_id
            )
            
            augmented_datasets.append(augmented_ds)
            logger.info(f"  ✓ Created augmented dataset: {temp_repo_id}")
            logger.info(f"    Episodes: {augmented_ds.meta.total_episodes}")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to create augmented dataset: {e}")
            raise

    # Merge all augmented datasets
    logger.info(f"\nMerging {len(augmented_datasets)} augmented datasets...")
    
    try:
        final_dataset = merge_datasets(
            augmented_datasets,
            output_repo_id=output_repo_id
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Successfully created augmented dataset!")
        logger.info(f"{'='*60}")
        logger.info(f"Output: {output_repo_id}")
        logger.info(f"Total episodes: {final_dataset.meta.total_episodes}")
        logger.info(f"Original episodes: {original_dataset.meta.total_episodes}")
        logger.info(f"Augmentation factor: {final_dataset.meta.total_episodes / original_dataset.meta.total_episodes:.1f}x")
        logger.info(f"Total frames: {final_dataset.meta.total_frames}")
        logger.info(f"\nInstructions used:")
        for i, inst in enumerate(instructions):
            logger.info(f"  {i+1}. {inst}")
        
        if push_to_hub:
            logger.info(f"\nPushing to Hugging Face Hub: {output_repo_id}")
            try:
                final_dataset.push_to_hub(output_repo_id)
                logger.info(f"✓ Successfully pushed to hub!")
            except Exception as e:
                logger.warning(f"Failed to push to hub: {e}")
                logger.warning("Dataset is saved locally but not uploaded.")
        
        return final_dataset
        
    except Exception as e:
        logger.error(f"Failed to merge datasets: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Augment a robotics dataset with multiple language instruction variations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  lerobot-dataset-aug --dataset.repo_id="roboseasy/Pick_up_a_sock_and_place_it_in_the_basket"
  
  # Using instruction file
  lerobot-dataset-aug \\
      --dataset.repo_id="roboseasy/Pick_up_a_sock_and_place_it_in_the_basket" \\
      --instructions_file="instructions.txt"
  
  # Direct instructions
  lerobot-dataset-aug \\
      --dataset.repo_id="roboseasy/Pick_up_a_sock_and_place_it_in_the_basket" \\
      --instructions "pick the socks and place it into box" \\
      --instructions "grab a sock and put it in the basket"
        """
    )
    
    parser.add_argument(
        "--dataset.repo_id",
        dest="dataset_repo_id",
        type=str,
        required=True,
        help="Repository ID of the dataset to augment (e.g., 'roboseasy/Pick_up_a_sock_and_place_it_in_the_basket')"
    )
    
    parser.add_argument(
        "--instructions",
        action="append",
        type=str,
        help="Language instruction for augmentation (can be specified multiple times)"
    )
    
    parser.add_argument(
        "--instructions_file",
        type=str,
        help="Path to file containing instructions (one per line). Has priority over --instructions."
    )
    
    parser.add_argument(
        "--output_repo_id",
        type=str,
        help="Output repository ID (default: input_repo_id + '_AUG')"
    )
    
    parser.add_argument(
        "--no_push",
        action="store_true",
        help="Do not push the augmented dataset to Hugging Face Hub"
    )
    
    args = parser.parse_args()
    
    # Determine instructions source (priority: file > CLI > interactive)
    instructions = None
    instructions_file_path = "instructions.txt"
    
    if args.instructions_file:
        # Priority 1: Read from file
        instructions = read_instructions_from_file(args.instructions_file)
        instructions_file_path = args.instructions_file
    elif args.instructions:
        # Priority 2: Use CLI arguments
        instructions = args.instructions
        logger.info(f"Using {len(instructions)} instructions from command line")
    else:
        # Priority 3: Interactive mode
        instructions = get_instructions_interactive()
        # Save to file
        save_instructions_to_file(instructions, instructions_file_path)
    
    # Run augmentation
    augment_dataset(
        base_repo_id=args.dataset_repo_id,
        instructions=instructions,
        output_repo_id=args.output_repo_id,
        push_to_hub=not args.no_push
    )


if __name__ == "__main__":
    main()
