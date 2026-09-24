# MiniCPM-o Reference Audio

On the speech pipeline, pass an explicit speaker reference in
`audio.ref_audio` on `/v1/chat/completions`:

```python
import base64
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="unused")
reference = base64.b64encode(Path("reference.wav").read_bytes()).decode("ascii")
response = client.chat.completions.create(
    model="MiniCPM-o-4_5",
    messages=[{"role": "user", "content": "Please say hello."}],
    modalities=["text", "audio"],
    audio={
        "format": "wav",
        "ref_audio": f"data:audio/wav;base64,{reference}",
    },
)
```

`stage_params.code2wav.ref_audio` is an alternative, with higher priority than
`audio.ref_audio`. Both accept `prompt_wav` as an alias. The Python pipeline
client can also supply `ref_audio` through `extra_params`. References must be
base64 audio data URIs, inline `{data, media_type}` descriptors, or encoded audio
bytes for the Python client. Paths and HTTP URLs are not fetched by this stage;
read or download the file on the client before sending it.

The reference conditions Token2wav's speaker embedding, prompt tokens, and mel
features. Audio supplied in chat messages remains understanding input and is not
automatically used as the speaker reference. Without an explicit reference,
Token2wav uses the checkpoint's `assets/HT_ref_audio.wav` when available.

The vocoder caches only the most recently used reference by audio content. A
different reference, including switching back to the default, rebuilds the
conditioning. Invalid references fail instead of silently using the default.
Audio output remains non-streaming.

## Speaking text already known to the caller

For a text-only speech request whose exact words are known in advance, pass
`known_tts_text` to `/v1/chat/completions`. MiniCPM-o conditions its Talker on
hidden states from one Thinker prefill of those words, instead of generating
the same words one token at a time. The option is explicit: ordinary chat,
audio understanding, and video requests keep their normal generation path.

```python
response = client.chat.completions.create(
    model="MiniCPM-o-4_5",
    messages=[{"role": "user", "content": "Please say hello."}],
    modalities=["text", "audio"],
    extra_body={"known_tts_text": "Hello!"},
)
```

The caller is responsible for the text. This is not a speculative verification
of what autoregressive decoding would have produced; the resulting audio can
therefore differ from an ordinary chat response. The option requires the speech
pipeline and currently supports only non-streaming, text-only input. Each such
request is isolated from prefix reuse because the Talker needs every text
position's hidden state.
