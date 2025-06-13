"""
Preprocess the Audio-Logic-Reasoning dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./mmdata/alr_preprocessed")
    parser.add_argument("--load_dir", default="./mmdata/alr")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = args.load_dir

    dataset = datasets.load_dataset(
        path="alr_dataset.py",
        data_dir=data_source,
    )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    dev_dataset = dataset["dev"]

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"Regardless of whether your final output is text or audio, your final Answer output will be in the form 'Answer: entailed' or 'Answer: not-entailed'. "
        r"Any output that does not end there is disqualified."
        r"The reasoning process MUST BE enclosed within <think> </think> tags."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            text_question = example.pop("user_content")
            prompt = text_question + " " + instruction_following
            text_answer = example.pop("answer")
            audio_question = example.pop("audio_question")
            audio_answer = example.pop("audio_answer")
            ori_idx = example.pop("id")

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text",
                             "text": "You are Qwen, a virtual human, capable of perceiving auditory inputs, as well as generating text and speech."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": audio_question},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "audio_question": audio_question,
                "audio_answer": audio_answer,
                "ability": "logic",
                "reward_model": {"style": "rule", "ground_truth": text_answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "ori_idx": ori_idx,
                    "test_answer": text_answer,
                    "audio_answer": audio_answer,
                    "text_question": text_question,
                    "audio_question": audio_question,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)
    dev_dataset = dev_dataset.map(function=make_map_fn("dev"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    dev_dataset.to_parquet(os.path.join(local_dir, "dev.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
