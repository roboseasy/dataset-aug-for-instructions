

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
```