# MiniCPM-o

[MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) is a Gemini 2.5 Flash level model for vision, speech, and full-duplex live streaming. SGLang-Omni serves it in two ways:

| Mode | Endpoint | Use it for |
|---|---|---|
| Chat | `/v1/chat/completions` | One request, one reply, in text or speech |
| Full duplex | `/v1/realtime` (WebSocket) | A live voice or video call where the model listens and speaks at the same time |

## Prerequisites

Follow [Installation](../get_started/installation.md), then run from the repository root.

Chat with speech output:

```bash
python -m sglang_omni.cli serve --model-path openbmb/MiniCPM-o-4_5 --port 8000
```

Chat with text output only:

```bash
python -m sglang_omni.cli serve --model-path openbmb/MiniCPM-o-4_5 --text-only --port 8000
```

## Normal Usage


```python
import base64
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reference = base64.b64encode(Path("docs/_static/audio/male-voice.wav").read_bytes()).decode("ascii")
response = client.chat.completions.create(
    model="MiniCPM-o-4_5",
    messages=[{"role": "user", "content": "Please say hello."}],
    modalities=["text", "audio"],
    audio={
        "format": "wav",
        "ref_audio": f"data:audio/wav;base64,{reference}",
    },
)
message = response.choices[0].message
Path("reply.wav").write_bytes(base64.b64decode(message.audio.data))
print(message.audio.transcript or message.content)
```

## Video Chat Inputs

Chat accepts inline video_url content and top-level video or videos fields.
Top-level requests without use_audio_in_video retain the existing shared video
sampling and resizing policy. They place explicit images, video frames, and
independent audio before the final user message's text, in that order.
Predecoded video tensors remain supported, including mixed file/tensor inputs.

Inline videos and top-level requests with use_audio_in_video use MiniCPM timed
video sampling. Set use_audio on an inline video, or use_audio_in_video on a
top-level request, to include interleaved audio from file or URL sources.
Top-level interleaved videos follow explicit images and independent audio.

Timed video frames retain native resolution by default; the MiniCPM image processor
performs the final resize. video_max_frames limits the sampled frame count,
while video_max_pixels and video_total_pixels resize frames after decoding.
Those pixel options do not bound the initial native-resolution allocation.

## Native Full-Duplex Ability

MiniCPM-o 4.5 also serves live full-duplex voice and video calls on `/v1/realtime`. Launching, settings, the browser demo and the protocol are in [Full Duplex Interactive Model](../developer_reference/full_duplex_interactive_model.md#native-full-duplex-minicpm-o).

## Text to speech

`/v1/audio/speech` reads the given text aloud. MiniCPM-o prefills the text in one Thinker pass and conditions the Talker on each of its tokens, instead of generating the same words one token at a time. Set `language` to `Chinese` for Chinese text, and pass a base64 audio data URI as `ref_audio` to clone a voice.

```python
audio = client.audio.speech.create(
    model="MiniCPM-o-4_5",
    voice="default",
    input="Hello!",
    extra_body={"language": "English"},
)
```

The speech output is non-streaming. The sampling fields you set, such as `temperature`, `top_p` and `max_new_tokens`, apply to the Talker, which generates the speech; the rest keep the Talker's defaults.

## Speech Stage Placement

The speech pipeline runs three processes on one GPU: the Thinker with the encoders, the Talker, and code2wav. They share the GPU through CUDA MPS (`mps: auto`, see [Same-GPU Data Parallelism with CUDA MPS](../basic_usage/mps_dp.md)), so their kernels run at the same time instead of taking turns, and code2wav's decode loop does not compete with the Talker's scheduler for one Python interpreter lock.

With 16 to 128 concurrent Seed-TTS voice-clone requests on an H200, this produces 21–43% more audio per second than code2wav in the Talker's process without MPS, at a lower p95 latency; on an H100 at 16 it produces 10% more. One request at a time it is 1% slower on an H200 and 9% slower on an H100. For single-stream use, put code2wav back in the Talker's process and turn MPS off:

```bash
python -m sglang_omni.cli serve --model-path openbmb/MiniCPM-o-4_5 --port 8000 \
  --code2wav.process talker --mps off
```
