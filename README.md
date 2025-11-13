

lerobot/src/lerobot/scripts/lerobot_dataset_aug.py

이 위치에 놓아주세요.

/home/khw/lerobot/pyproject.toml

에서 아래 내용을 넣어주세요.

[project.scripts]

이 위치에 

lerobot-dataset-aug="lerobot.scripts.lerobot_dataset_aug:main"

를 넣어주세요.

## 실행방법

```
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

```