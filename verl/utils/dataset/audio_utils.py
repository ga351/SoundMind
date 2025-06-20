import base64
from io import BytesIO

import audioread
import av
import librosa
import numpy as np


def _check_if_video_has_audio(video_path):
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True


def process_audio_info(conversations: list[dict] | list[list[dict]]):
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele:
                        path = ele["audio"]
                        if isinstance(path, np.ndarray):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")
                            audios.append(path)
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = base64.b64decode(base64_data)
                            audios.append(librosa.load(BytesIO(data), sr=16000)[0])
                        elif path.startswith("http://") or path.startswith("https://"):
                            audios.append(librosa.load(audioread.ffdec.FFmpegAudioFile(path), sr=16000)[0])
                        elif path.startswith("file://"):
                            audios.append(librosa.load(path[len("file://") :], sr=16000)[0])
                        else:
                            audios.append(librosa.load(path, sr=16000)[0])
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
    if len(audios) == 0:
        audios = None
    return audios
