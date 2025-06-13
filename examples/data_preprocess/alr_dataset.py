import os
import json
import datasets
from datasets import DatasetInfo, GeneratorBasedBuilder, SplitGenerator, Split, Value, Audio, Features

class ALRDataset(GeneratorBasedBuilder):
    """Custom dataset loader for Audio-Logic-Reasoning (ALR) dataset."""

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description="Audio-Logic-Reasoning multimodal dataset with text and audio.",
            features=Features({
                "id": Value("string"),
                "user_content": Value("string"),   # Text corresponding to question
                "answer": Value("string"),         # Text corresponding to answer
                "label": Value("int32"),           # 0 or 1
                "audio_question": Audio(sampling_rate=16_000),
                "audio_answer": Audio(sampling_rate=16_000),
            }),
        )

    def _split_generators(self, dl_manager):
        base_path = self.config.data_dir or "./mmdata/alr"

        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"data_dir": base_path, "jsonl_file": "train.jsonl", "split": "train"}),
            SplitGenerator(name=Split.VALIDATION, gen_kwargs={"data_dir": base_path, "jsonl_file": "dev.jsonl", "split": "dev"}),
            SplitGenerator(name=Split.TEST, gen_kwargs={"data_dir": base_path, "jsonl_file": "test.jsonl", "split": "test"}),
        ]

    def _generate_examples(self, data_dir, jsonl_file, split):
        jsonl_path = os.path.join(data_dir, jsonl_file)
        audio_root = os.path.join(data_dir, split)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                item = json.loads(line.strip())
                sample_id = item["id"]
                folder_path = os.path.join(audio_root, sample_id)

                question_audio_path = os.path.join(folder_path, "question.wav")
                label = int(item["label"])
                answer_audio_path = os.path.join(folder_path, f"answer_{label}.wav")

                yield idx, {
                    "id": sample_id,
                    "user_content": item["user_content"],
                    "answer": item["answer"],
                    "label": label,
                    "audio_question": question_audio_path,
                    "audio_answer": answer_audio_path,
                }
