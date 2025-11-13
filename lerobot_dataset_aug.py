#!/usr/bin/env python

"""
Unified Dataset Augmentation & Merge Script for Language Instructions

사용 예시:
    # 인터랙티브 모드
    lerobot-dataset-aug \
        --dataset_load_repo_id="roboseasy/Pick_up_a_sock_and_place_it_in_the_basket" \
        --dataset_output_repo_id="roboseasy/Pick_up_a_sock_and_place_it_in_the_basket_AUG"

    # 지시어 파일 사용
    lerobot-dataset-aug \
        --dataset_load_repo_id="roboseasy/Pick_up_a_sock_and_place_it_in_the_basket" \
        --dataset_output_repo_id="roboseasy/Pick_up_a_sock_and_place_it_in_the_basket_AUG" \
        --instructions_file="instructions.txt"

    # 명령줄 직접 입력
    lerobot-dataset-aug \
        --dataset_load_repo_id="roboseasy/pick_and_place_1" \
        --dataset_output_repo_id="roboseasy/pick_and_place_aug_test" \
        --instructions "Pick up the sock and put it in the basket" \
        --instructions "Grab the sock and place it into the basket" \
        --instructions "Task the sock and put it into the basket" \
        --push_to_hub
        
        
python -c "import pandas as pd; df = pd.read_parquet('/home/khw/.cache/huggingface/lerobot/roboseasy/pick_and_place_aug_test/meta/tasks.parquet'); print(df)"

"""

import argparse
import os
import sys
import shutil
from tqdm import tqdm

# test-duplicate.py와 merge_datasets_simple.py의 함수 직접 import
import importlib.util

def import_from_path(module_path, func_name):
    spec = importlib.util.spec_from_file_location("mod", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)

duplicate_dataset = import_from_path(
    os.path.join(os.path.dirname(__file__), "test-duplicate.py"),
    "duplicate_dataset"
)
merge_datasets = import_from_path(
    os.path.join(os.path.dirname(__file__), "merge_datasets_simple.py"),
    "merge_datasets"
)

def get_dataset_path(repo_id):
    from pathlib import Path
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id

def parse_args():
    parser = argparse.ArgumentParser(description="Augment dataset with multiple language instructions.")
    parser.add_argument("--dataset_load_repo_id", type=str, required=True, help="Source dataset repo_id (or path)")
    parser.add_argument("--dataset_output_repo_id", type=str, required=True, help="Output dataset repo_id (or path)")
    parser.add_argument("--instructions", type=str, action="append", help="Instruction string (can be used multiple times)")
    parser.add_argument("--instructions_file", type=str, help="Path to file with instructions (one per line)")
    parser.add_argument("--push_to_hub", action="store_true", help="If set, push merged dataset to HuggingFace hub")
    return parser.parse_args()

def interactive_instructions():
    print("새로운 지시어를 정해주세요 (엔터로 입력, esc 또는 빈 줄로 종료):")
    instructions = []
    try:
        while True:
            line = input("> ").strip()
            if not line:
                print("입력 종료.")
                break
            instructions.append(line)
            print("다음 지시어를 정해주세요 (계속하려면 엔터, 끝내려면 esc 또는 빈 줄):")
    except (KeyboardInterrupt, EOFError):
        print("\n입력 종료.")
    return instructions

def load_instructions(args):
    # 명령줄 입력
    if args.instructions:
        return args.instructions
    # 파일 입력
    if args.instructions_file:
        if not os.path.exists(args.instructions_file):
            print("지시어 파일이 없습니다.")
            sys.exit(1)
        with open(args.instructions_file, "r") as f:
            instructions = [line.strip() for line in f if line.strip()]
        if not instructions:
            print("지시어 파일에 유효한 지시어가 없습니다.")
            sys.exit(1)
        return instructions
    # 인터랙티브 입력
    return interactive_instructions()

def main():
    args = parse_args()
    instructions = load_instructions(args)
    if not instructions:
        print("지시어가 입력되지 않았습니다. 종료합니다.")
        sys.exit(1)

    # 1. instruction별 복제본 생성 (test-duplicate.py 활용)
    inst_repo_ids = []
    for idx, instruction in enumerate(tqdm(instructions, desc="지시어별 복제")):
        inst_repo_id = f"{args.dataset_output_repo_id}_inst{idx}"
        duplicate_dataset(
            src_repo_id=args.dataset_load_repo_id,
            dst_repo_id=inst_repo_id,
            single_task=instruction
        )
        inst_repo_ids.append(inst_repo_id)

    # 2. 병합 및 업로드 (merge_datasets_simple.py 활용)
    merge_datasets(
        repo_ids=inst_repo_ids,
        output_repo_id=args.dataset_output_repo_id,
        push_to_hub=args.push_to_hub
    )

    # 3. 임시 복제본 자동 삭제(clean-up)
    for inst_repo_id in inst_repo_ids:
        inst_path = get_dataset_path(inst_repo_id)
        if inst_path.exists():
            try:
                shutil.rmtree(inst_path)
                print(f"임시 복제본 삭제 완료: {inst_path}")
            except Exception as e:
                print(f"임시 복제본 삭제 실패: {inst_path} ({e})")

    print("Augmentation and merge complete.")
    print("  Output repo_id:", args.dataset_output_repo_id)
    print("  Instructions:", instructions)
    if args.push_to_hub:
        print("  (허깅페이스 허브에 업로드 완료)")

if __name__ == "__main__":
    main()
