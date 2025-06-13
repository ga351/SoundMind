
import re
import struct
import io
from typing import Dict, Tuple, Optional, Union, List

import numpy as np
import librosa
import webrtcvad
import soundfile as sf


def extract_solution(response_text: str) -> Tuple[Optional[str], str]:
    """Extract the final answer from the assistant's structured response.

    This function identifies the assistant's response section and extracts content inside <answer> tags,
    making sure to correctly handle spacing, nested structures, and escaped characters, while also retaining
    the complete response body for further validation.
    """
    assistant_delimiters = ["<|im_start|>assistant", "Assistant:"]
    content = None

    for delimiter in assistant_delimiters:
        if delimiter in response_text:
            content = response_text.split(delimiter, 1)[1]
            break

    if content is None:
        print("[extract_solution] Assistant response delimiter not found.")
        return None, response_text

    content = content.strip().replace("\r", "").replace("\t", " ")

    answer_matches = list(re.finditer(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE))

    if not answer_matches:
        print("[extract_solution] No <answer> tags found.")
        return None, content

    answer = answer_matches[-1].group(1).strip()
    return answer, content


def parse_solution_text_format(reference_solution_text: str) -> Dict[str, str]:
    """
    Extracts the final entailment label from the human-annotated reasoning.
    Designed specifically for Logi-QA 2.0 style tasks in the ALR dataset.

    Args:
        reference_solution_text: Full reasoning + conclusion annotation in text.

    Returns:
        A dictionary with {'label': 'entailed'} or {'label': 'not-entailed'}
    """

    lowered = reference_solution_text.lower()
    if 'the conclusion is entailed' in lowered or 'entailed' in lowered:
        return {'label': 'entailed'}
    elif 'the conclusion is not entailed' in lowered or 'not-entailed' in lowered:
        return {'label': 'not-entailed'}
    else:
        return {'label': 'unknown'}



def parse_model_answer(answer_text: str, expected_fields: list = None) -> Optional[Dict[str, str]]:
    """
    Extracts the final entailment decision from the model's <answer> block.
    Supports flexible NLI phrasing, such as 'the conclusion must be true' or 'is entailed'.

    Args:
        answer_text: The text inside <answer>...</answer> section.
        expected_fields: Ignored in NLI tasks.

    Returns:
        A dictionary with {'label': 'entailed'} or {'label': 'not-entailed'}, or None if undecidable.
    """

    normalized = answer_text.lower().strip()

    # Direct phrasing
    if 'entailed' in normalized and 'not entailed' not in normalized:
        return {'label': 'entailed'}
    elif 'not entailed' in normalized or 'not-entailed' in normalized:
        return {'label': 'not-entailed'}

    # Pattern-based fallback (e.g., conclusion must be true or cannot be concluded)
    if re.search(r'the conclusion.*(must be true|is true)', normalized):
        return {'label': 'entailed'}
    if re.search(r'the conclusion.*(is not true|cannot be.*concluded|not necessarily true)', normalized):
        return {'label': 'not-entailed'}
    return None


# def parse_solution_text_format(reference_text: str) -> Dict[str, str]:
#     """
#     Extract the gold-standard entailment label from human-written reference reasoning text.
#
#     This function is specifically designed for the ALR dataset's logical inference task based on Logi-QA 2.0.
#     It supports natural variations in how 'entailed' and 'not-entailed' are expressed, with a mix of
#     deterministic regex patterns and context-aware heuristics. It prioritizes precision over recall to reduce noise.
#     """
#
#     text = reference_text.lower()
#     text = text.replace('\n', ' ').replace('\t', ' ')
#     text = re.sub(r'\s+', ' ', text)
#
#     entailed_phrases = [
#         r"the conclusion is (logically )?entailed",
#         r"this implies the conclusion",
#         r"we (can|could) conclude that",
#         r"therefore,? the conclusion (must|has to|does) (follow|be true)",
#         r"the conclusion follows from (the )?premises",
#         r"it is clear that the conclusion is true",
#     ]
#
#     not_entailed_phrases = [
#         r"the conclusion is not (logically )?entailed",
#         r"we cannot (logically )?conclude",
#         r"the conclusion does not follow",
#         r"the premises do not support the conclusion",
#         r"there is not enough information",
#         r"the conclusion might not be true",
#         r"it is not necessarily true",
#         r"the conclusion (could|may|might) be false",
#     ]
#
#     for pattern in not_entailed_phrases:
#         if re.search(pattern, text):
#             return {"label": "not-entailed"}
#
#     for pattern in entailed_phrases:
#         if re.search(pattern, text):
#             return {"label": "entailed"}
#     return {"label": "unknown"}
#
#
# def parse_model_answer(answer_text: str, expected_fields: list = None) -> Optional[Dict[str, str]]:
#     """
#     Analyze model's answer text to infer whether it claims the conclusion is entailed or not.
#     The function parses nuanced natural language using overlapping positive and negative regex matches,
#     contextual cues, and logical negation disambiguation to avoid misclassification.
#
#     Returns:
#         A dictionary of form {'label': 'entailed'} or {'label': 'not-entailed'}, or None if undecidable.
#     """
#
#     text = answer_text.strip().lower()
#     text = re.sub(r'\s+', ' ', text)
#
#     entailed_patterns = [
#         r"the conclusion is (logically )?entailed",
#         r"this (definitely|clearly) implies the conclusion",
#         r"we (can|could) conclude that",
#         r"the conclusion follows",
#         r"it must be true",
#         r"the answer is entailed",
#     ]
#
#     not_entailed_patterns = [
#         r"the conclusion is not (logically )?entailed",
#         r"we cannot (logically )?conclude",
#         r"the conclusion does not follow",
#         r"there is insufficient information",
#         r"the answer is not entailed",
#         r"it might not be true",
#         r"it is not necessarily true",
#         r"no clear conclusion can be drawn",
#     ]
#
#     for neg in not_entailed_patterns:
#         if re.search(neg, text):
#             return {"label": "not-entailed"}
#
#     for pos in entailed_patterns:
#         if re.search(pos, text):
#             return {"label": "entailed"}
#
#     # Handle ambiguous negations, e.g., "it is entailed" vs "not necessarily entailed"
#     if "entailed" in text:
#         if re.search(r"not.*entailed", text):
#             return {"label": "not-entailed"}
#         else:
#             return {"label": "entailed"}
#     return None




def validate_response_structure(response_body: str) -> bool:
    """Checks whether the model's response contains the correct reasoning-answer structure.

    Ensures presence and correct ordering of the <think> and <answer> sections and validates structure compliance
    such as non-repetition, non-nesting, and character constraints around anchor tokens like "Answer:".
    """
    print("\n[validate_response_structure] Checking tag and anchor consistency...")

    required_tags = {
        'reason_start': '<think>',
        'reason_end': '</think>',
        'answer_start': '<answer>',
        'answer_end': '</answer>',
    }

    for tag_name, tag in required_tags.items():
        count = response_body.count(tag)
        print(f"  {tag}: count = {count}")
        if count != 1:
            print(f"  ✗ Structural error: {tag} appears {count} times.")
            return False

    tag_order = [response_body.find(tag) for tag in required_tags.values()]
    if not all(x < y for x, y in zip(tag_order, tag_order[1:])):
        print("  ✗ Tag order is incorrect. Required: <think>...</think><answer>...</answer>")
        return False

    trimmed = re.sub(r'\s+', '', response_body.lower())
    if not trimmed.endswith("answer:") and "answer:" not in trimmed[-10:]:
        print("  ✗ 'Answer:' keyword not found at the end.")
        return False

    print("  ✓ Structure is valid.")
    return True


def get_reasoning_audio_duration(
    audio_input: Union[str, bytes, np.ndarray],
    sr: int = 16000,
    vad_mode: int = 3,
    frame_duration_ms: int = 30,
    reasoning_ratio_threshold: float = 0.85
) -> float:
    """Estimates the reasoning duration in a model's audio output using VAD and segment heuristics.

    Voice activity detection is applied to detect speech frames. Reasoning segments are assumed to occur
    in the first portion of the audio before the answer is announced. The function returns the estimated
    reasoning duration in seconds.
    """

    if isinstance(audio_input, str):
        y, orig_sr = librosa.load(audio_input, sr=sr)
    elif isinstance(audio_input, bytes):
        with io.BytesIO(audio_input) as bio:
            y, orig_sr = sf.read(bio, dtype='float32')
            if orig_sr != sr:
                y = librosa.resample(y, orig_sr, sr)
    elif isinstance(audio_input, np.ndarray):
        y = librosa.resample(audio_input, orig_sr=sr, target_sr=sr) if sr != 16000 else audio_input
    else:
        raise ValueError("Unsupported audio input type.")

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    y = y / np.max(np.abs(y))

    y_pcm = (y * 32767).astype(np.int16)
    pcm_bytes = struct.pack(f'{len(y_pcm)}h', *y_pcm)

    vad = webrtcvad.Vad(vad_mode)
    frame_len = int(sr * frame_duration_ms / 1000)
    num_frames = len(y_pcm) // frame_len

    voiced_segments = []
    for i in range(num_frames):
        start = i * frame_len
        end = start + frame_len
        if end > len(y_pcm):
            break
        frame = pcm_bytes[start * 2:end * 2]
        if vad.is_speech(frame, sr):
            voiced_segments.append((start, end))

    if not voiced_segments:
        return 0.0

    merged = []
    last_start, last_end = voiced_segments[0]
    for start, end in voiced_segments[1:]:
        if start - last_end < sr * 0.3:
            last_end = end
        else:
            merged.append((last_start, last_end))
            last_start, last_end = start, end
    merged.append((last_start, last_end))

    total_samples = len(y)
    cutoff_sample = int(total_samples * reasoning_ratio_threshold)
    reasoning_segments = [(s, e) for s, e in merged if e < cutoff_sample]
    reasoning_samples = sum(e - s for s, e in reasoning_segments)
    duration_seconds = reasoning_samples / sr
    return duration_seconds


def compute_score(
    solution_str: str,
    ground_truth: Dict[str, str],
    format_text_bonus: float = 1.0,
    format_audio_bonus: float = 0.5,
    correct_answer_reward: float = 2.0,
    text_length_weight: float = 1.0,
    audio_length_weight: float = 0.75,
    ref_text_token_count: Optional[int] = None,
    ref_audio_duration: Optional[float] = None,
    model_audio_input: Optional[Union[str, bytes, np.ndarray]] = None
):

    gt_text = ground_truth.get('solution_text_format', '')
    gt_status = parse_solution_text_format(gt_text)
    expected_names = list(gt_status.keys())

    answer_content, response_body = extract_solution(solution_str)

    format_valid = validate_response_structure(response_body)
    last_segment = response_body.strip()[-20:].lower().replace(' ', '')
    ends_with_answer_tag = 'answer:' in last_segment[-5:]
    text_format_score = format_text_bonus if format_valid and ends_with_answer_tag else 0

    has_audio_tags = '<speak>' in response_body.lower() and '</speak>' in response_body.lower()
    audio_format_score = format_audio_bonus if has_audio_tags else 0

    answer_score = 0
    if format_valid and answer_content:
        pred_status = parse_model_answer(answer_content, expected_names)
        if pred_status and pred_status == gt_status:
            answer_score = correct_answer_reward

    model_token_count = len(response_body.split())
    length_text_score = 0
    if ref_text_token_count and ref_text_token_count > 0:
        length_ratio = model_token_count / ref_text_token_count
        length_text_score = text_length_weight * min(1.0, length_ratio)

    model_audio_duration = None
    length_audio_score = 0
    if model_audio_input:
        model_audio_duration = get_reasoning_audio_duration(model_audio_input)
    if ref_audio_duration and ref_audio_duration > 0 and model_audio_duration is not None:
        audio_ratio = model_audio_duration / ref_audio_duration
        length_audio_score = audio_length_weight * min(1.0, audio_ratio)

    total_score = (
        text_format_score +
        audio_format_score +
        answer_score +
        length_text_score +
        length_audio_score
    )
    return total_score
