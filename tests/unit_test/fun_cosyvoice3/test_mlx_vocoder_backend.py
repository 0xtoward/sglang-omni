# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import threading
import weakref

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_flatten  # noqa: E402

from sglang_omni.models.fun_cosyvoice3.mlx.vocoder import (  # noqa: E402
    FunCosyVoice3MlxVocoder,
)
from sglang_omni.models.fun_cosyvoice3.mlx.vocoder.compiled_dit import (  # noqa: E402
    CompiledDiT,
    RotaryEstimator,
)
from sglang_omni.models.fun_cosyvoice3.mlx.vocoder.config import (  # noqa: E402
    FlowConfig,
    HiFTConfig,
)
from sglang_omni.models.fun_cosyvoice3.mlx.vocoder.dit import (  # noqa: E402
    Attention,
    DiT,
    affine_modulation,
    gated_residual,
    layer_norm,
)
from sglang_omni.models.fun_cosyvoice3.mlx.vocoder.flow import (  # noqa: E402
    CausalMaskedDiffWithDiT,
)
from sglang_omni.models.fun_cosyvoice3.mlx.vocoder.flow_matching import (  # noqa: E402
    CausalConditionalCFM,
)
from sglang_omni.models.fun_cosyvoice3.mlx.vocoder.hift import (  # noqa: E402
    CausalHiFTGenerator,
)
from sglang_omni.models.fun_cosyvoice3.mlx.vocoder.loader import (  # noqa: E402
    map_flow_weight,
    map_hift_weight,
)


def _write_tiny_artifact(tmp_path, *, hift_prefix="hifigan"):
    flow_config = FlowConfig(
        input_size=4,
        output_size=4,
        spk_embed_dim=3,
        vocab_size=16,
        pre_lookahead_len=1,
        pre_lookahead_channels=16,
        token_mel_ratio=2,
        dit_hidden_size=16,
        dit_depth=1,
        dit_num_heads=2,
        dit_head_dim=8,
        dit_mlp_ratio=2.0,
        dit_mel_dim=4,
        dit_mu_dim=4,
        dit_spk_dim=4,
        dit_static_chunk_size=4,
        n_timesteps=1,
    )
    hift_config = HiFTConfig(
        in_channels=4,
        base_channels=8,
        nb_harmonics=1,
        sampling_rate=24000,
        upsample_rates=[2],
        upsample_kernel_sizes=[4],
        istft_params={"n_fft": 4, "hop_len": 2},
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1]],
        source_resblock_kernel_sizes=[3],
        source_resblock_dilation_sizes=[[1]],
        conv_pre_look_right=1,
    )
    flow = CausalMaskedDiffWithDiT(flow_config)
    hift = CausalHiFTGenerator(hift_config)
    weights = {
        **{f"flow.{name}": value for name, value in tree_flatten(flow.parameters())},
        **{
            f"{hift_prefix}.{name}": value
            for name, value in tree_flatten(hift.parameters())
        },
    }
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "cosyvoice3",
                "flow": {
                    "input_size": 4,
                    "output_size": 4,
                    "spk_embed_dim": 3,
                    "vocab_size": 16,
                    "pre_lookahead_len": 1,
                    "pre_lookahead_channels": 16,
                    "token_mel_ratio": 2,
                    "n_timesteps": 1,
                    "dit": {
                        "dim": 16,
                        "depth": 1,
                        "heads": 2,
                        "dim_head": 8,
                        "ff_mult": 2.0,
                        "mel_dim": 4,
                        "mu_dim": 4,
                        "spk_dim": 4,
                        "static_chunk_size": 4,
                    },
                },
                "hifigan": {
                    "in_channels": 4,
                    "base_channels": 8,
                    "nb_harmonics": 1,
                    "sampling_rate": 24000,
                    "upsample_rates": [2],
                    "upsample_kernel_sizes": [4],
                    "istft_n_fft": 4,
                    "istft_hop_len": 2,
                    "resblock_kernel_sizes": [3],
                    "resblock_dilation_sizes": [[1]],
                    "source_resblock_kernel_sizes": [3],
                    "source_resblock_dilation_sizes": [[1]],
                    "conv_pre_look_right": 1,
                },
            }
        ),
        encoding="utf-8",
    )


def test_plus_artifact_weight_key_mapping():
    assert (
        map_flow_weight("flow.decoder.estimator.transformer_blocks.0.ff.ff_0_0.weight")
        == "decoder.estimator.transformer_blocks.0.ff.ff.0.weight"
    )
    assert (
        map_hift_weight("hifigan.f0_predictor.condnet_4.conv.weight")
        == "f0_predictor.condnet.2.weight"
    )
    assert (
        map_hift_weight("hifigan.resblocks.0.convs1.0.conv.weight")
        == "resblocks.0.convs1.0.weight"
    )
    assert (
        map_hift_weight("hift.resblocks.0.convs1.0.weight")
        == "resblocks.0.convs1.0.weight"
    )


@pytest.mark.parametrize("batch_size,mel_frames", [(1, 5), (2, 9)])
@pytest.mark.parametrize(
    "activation_dtype,modulation_dtype",
    [(mx.float16, mx.float32), (mx.float32, mx.float32), (mx.float16, mx.float16)],
)
@pytest.mark.parametrize("is_final", [False, True])
def test_dit_pointwise_fusion_preserves_views_and_fresh_inputs(
    batch_size: int,
    mel_frames: int,
    activation_dtype: mx.Dtype,
    modulation_dtype: mx.Dtype,
    is_final: bool,
) -> None:
    channel_count = 8
    for random_seed in (7, 19):
        mx.random.seed(random_seed)
        residual = (
            mx.random.normal((batch_size, channel_count, mel_frames))
            .astype(activation_dtype)
            .transpose(0, 2, 1)
        )
        branch = mx.random.normal(residual.shape).astype(modulation_dtype)
        modulation = mx.random.normal((batch_size, channel_count * 3)).astype(
            modulation_dtype
        )
        scale, shift, gate = mx.split(modulation, 3, axis=-1)
        mx.eval(residual, branch, scale, shift, gate)
        if is_final:
            expected_affine = residual * (1 + scale)[:, None, :] + shift[:, None, :]
        else:
            expected_affine = residual * (1 + scale[:, None]) + shift[:, None]
        expected_residual = residual + gate[:, None] * branch
        actual_affine = affine_modulation(residual, scale, shift, is_final)
        actual_residual = gated_residual(residual, gate, branch)
        mx.eval(expected_affine, expected_residual, actual_affine, actual_residual)
        assert actual_affine.shape == residual.shape
        assert actual_residual.shape == residual.shape
        assert actual_affine.dtype == expected_affine.dtype
        assert actual_residual.dtype == expected_residual.dtype
        np.testing.assert_allclose(
            np.asarray(actual_affine), np.asarray(expected_affine), rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            np.asarray(actual_residual),
            np.asarray(expected_residual),
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.fixture
def tiny_dit() -> DiT:
    mx.random.seed(31)
    estimator = DiT(
        dim=32, depth=2, heads=2, dim_head=8, mel_dim=4, mu_dim=4, spk_dim=4
    )
    estimator.set_dtype(mx.float16)
    estimator.eval()
    mx.eval(estimator.parameters())
    return estimator


def dit_inputs(
    mel_frames: int, dtype: mx.Dtype
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    hidden_states = mx.random.normal((2, 4, mel_frames)).astype(dtype)
    mask = mx.ones((2, 1, mel_frames), dtype=dtype)
    conditioning = mx.random.normal(hidden_states.shape).astype(mx.float16)
    timestep = mx.full((2,), 0.25, dtype=dtype)
    speakers = mx.random.normal((2, 4)).astype(mx.float16)
    prompt = mx.random.normal(hidden_states.shape).astype(mx.float16)
    mx.eval(hidden_states, mask, conditioning, timestep, speakers, prompt)
    return hidden_states, mask, conditioning, timestep, speakers, prompt


def assert_dit_parity(
    native: DiT,
    compiled: CompiledDiT,
    inputs: tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array],
) -> np.ndarray:
    compiled.prepare()
    expected = native(*inputs)
    actual = compiled(*inputs)
    mx.eval(expected, actual)
    mx.synchronize()
    assert actual.shape == inputs[0].shape
    assert actual.dtype == expected.dtype
    actual_numpy = np.asarray(actual)
    assert np.isfinite(actual_numpy).all()
    np.testing.assert_allclose(actual_numpy, np.asarray(expected), rtol=1e-3, atol=1e-3)
    return actual_numpy.copy()


@pytest.mark.parametrize("activation_dtype", [mx.float16, mx.float32])
def test_compiled_dit_tracks_lengths_inputs_weights_and_rotary(
    tiny_dit: DiT, activation_dtype: mx.Dtype
) -> None:
    parameter_keys = [name for name, _ in tree_flatten(tiny_dit.parameters())]
    compiled = CompiledDiT(tiny_dit, cache_size=2)
    for mel_frames in (9, 5, 15, 9):
        inputs = dit_inputs(mel_frames, activation_dtype)
        assert_dit_parity(tiny_dit, compiled, inputs)
    previous = assert_dit_parity(tiny_dit, compiled, inputs)
    fresh_inputs = dit_inputs(9, activation_dtype)
    fresh_output = assert_dit_parity(tiny_dit, compiled, fresh_inputs)
    assert not np.allclose(previous, fresh_output)

    tiny_dit.update({"proj_out": {"bias": tiny_dit.proj_out.bias + 0.5}})
    updated_output = assert_dit_parity(tiny_dit, compiled, fresh_inputs)
    assert not np.allclose(fresh_output, updated_output)

    rotary = tiny_dit.rotary_embed
    rotary.cos = mx.zeros_like(rotary.cos)
    rotary.sin = mx.ones_like(rotary.sin)
    mx.eval(rotary.cos, rotary.sin)
    rotary_output = assert_dit_parity(tiny_dit, compiled, fresh_inputs)
    assert not np.allclose(updated_output, rotary_output)
    assert parameter_keys == [name for name, _ in tree_flatten(tiny_dit.parameters())]


def test_compiled_dit_lru_reuses_and_releases_closures(
    tiny_dit: DiT, monkeypatch: pytest.MonkeyPatch
) -> None:
    native_compile = mx.compile
    closures: list[weakref.ReferenceType[RotaryEstimator]] = []

    def tracking_compile(
        function: RotaryEstimator, *, inputs: DiT, shapeless: bool
    ) -> RotaryEstimator:
        closures.append(weakref.ref(function))
        return native_compile(function, inputs=inputs, shapeless=shapeless)

    monkeypatch.setattr(mx, "compile", tracking_compile)
    compiled = CompiledDiT(tiny_dit, cache_size=2)
    for mel_frames, dtype, expected_count in (
        (9, mx.float16, 1),
        (9, mx.float16, 1),
        (5, mx.float16, 2),
        (15, mx.float16, 3),
        (9, mx.float16, 4),
        (9, mx.float32, 5),
        (9, mx.float32, 5),
    ):
        assert_dit_parity(tiny_dit, compiled, dit_inputs(mel_frames, dtype))
        assert len(closures) == expected_count
        assert len(compiled.cache) <= 2
    assert closures[0]() is None
    assert closures[1]() is None
    assert closures[2]() is None

    tiny_dit.set_dtype(mx.float32)
    compiled.prepare()
    assert len(compiled.cache) == 0
    assert all(closure() is None for closure in closures)
    assert_dit_parity(tiny_dit, compiled, dit_inputs(9, mx.float32))
    assert len(closures) == 6


@pytest.mark.parametrize("step_count", [2, 3])
def test_compiled_cfm_preserves_euler_and_parameter_keys(
    tiny_dit: DiT, step_count: int
) -> None:
    solver = CausalConditionalCFM(tiny_dit)
    parameter_keys = [name for name, _ in tree_flatten(solver.parameters())]
    mu = mx.random.normal((1, 4, 9)).astype(mx.float16)
    mask = mx.ones((1, 1, 9), dtype=mx.float16)
    speakers = mx.random.normal((1, 4)).astype(mx.float16)
    prompt = mx.random.normal(mu.shape).astype(mx.float16)
    noise = mx.random.normal(mu.shape).astype(mx.float16)
    mx.eval(mu, mask, speakers, prompt, noise)
    expected = solver(mu, mask, speakers, prompt, step_count, noise=noise)
    mx.eval(expected)
    solver.enable_compile(cache_size=2)
    actual = solver(mu, mask, speakers, prompt, step_count, noise=noise)
    mx.eval(actual)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.isfinite(np.asarray(actual)).all()
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-3, atol=1e-3
    )
    compiled_estimator = solver.compiled_estimator
    for change_dtype in (False, True):
        if change_dtype:
            tiny_dit.set_dtype(mx.float32)
        else:
            tiny_dit.update({"proj_out": {"bias": tiny_dit.proj_out.bias + 0.5}})
        solver.compiled_estimator = None
        updated_expected = solver(mu, mask, speakers, prompt, step_count, noise=noise)
        mx.eval(updated_expected)
        solver.compiled_estimator = compiled_estimator
        updated_actual = solver(mu, mask, speakers, prompt, step_count, noise=noise)
        mx.eval(updated_actual)
        assert updated_actual.dtype == updated_expected.dtype
        np.testing.assert_allclose(
            np.asarray(updated_actual),
            np.asarray(updated_expected),
            rtol=1e-3,
            atol=1e-3,
        )
    assert not np.allclose(np.asarray(actual), np.asarray(updated_actual))
    assert parameter_keys == [name for name, _ in tree_flatten(solver.parameters())]


@pytest.mark.parametrize("cache_size", [0, -1])
def test_compiled_dit_rejects_nonpositive_cache_size(
    tiny_dit: DiT, cache_size: int
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        CompiledDiT(tiny_dit, cache_size)


def test_fused_attention_matches_explicit_reference():
    mx.random.seed(7)
    attention = Attention(dim=8, heads=2, dim_head=4)
    inputs = mx.random.normal((2, 5, 8))
    mask = mx.tril(mx.ones((2, 5, 5), dtype=mx.bool_))

    actual = attention(inputs, mask=mask)
    query = attention.to_q(inputs).reshape(2, 5, 2, 4).transpose(0, 2, 1, 3)
    key = attention.to_k(inputs).reshape(2, 5, 2, 4).transpose(0, 2, 1, 3)
    value = attention.to_v(inputs).reshape(2, 5, 2, 4).transpose(0, 2, 1, 3)
    scores = query @ key.transpose(0, 1, 3, 2) / math.sqrt(4)
    scores = mx.where(mask[:, None], scores, -float("inf"))
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
    expected = weights @ value
    expected = expected.transpose(0, 2, 1, 3).reshape(2, 5, 8)
    expected = attention.to_out[0](expected)

    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()


def test_fast_layer_norm_matches_reference_formula():
    mx.random.seed(11)
    inputs = mx.random.normal((2, 5, 16)).astype(mx.float16)

    actual = layer_norm(inputs)
    inputs_float = inputs.astype(mx.float32)
    mean = mx.mean(inputs_float, axis=-1, keepdims=True)
    variance = mx.var(inputs_float, axis=-1, keepdims=True)
    expected = ((inputs_float - mean) * mx.rsqrt(variance + 1e-6)).astype(inputs.dtype)

    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, rtol=1e-3, atol=1e-3).item()


def test_flow_noise_is_cast_to_model_dtype():
    class ZeroEstimator:
        out_channels = 4

        def __call__(self, x, *args):
            return mx.zeros_like(x)

    flow_matching = CausalConditionalCFM(ZeroEstimator())
    mu = mx.zeros((1, 4, 6), dtype=mx.float16)
    output = flow_matching(
        mu=mu,
        mask=mx.ones((1, 1, 6), dtype=mx.float16),
        spks=mx.zeros((1, 4), dtype=mx.float16),
        cond=mx.zeros_like(mu),
        n_timesteps=1,
    )

    mx.eval(output)
    assert flow_matching._rand_noise.dtype == mx.float32  # noqa: leading-underscore
    assert output.dtype == mx.float16


def test_load_and_decode_tiny_converted_artifact(tmp_path):
    _write_tiny_artifact(tmp_path)
    vocoder = FunCosyVoice3MlxVocoder.from_pretrained(str(tmp_path))

    waveform = vocoder.decode(
        token=[1, 2, 3],
        prompt_token=[4, 5],
        prompt_feat=np.zeros((4, 4), dtype=np.float32),
        embedding=np.ones(3, dtype=np.float32),
    )

    assert waveform.ndim == 1
    assert waveform.size > 0
    assert waveform.dtype == np.float32
    assert waveform.flags.c_contiguous
    assert np.isfinite(waveform).all()
    assert vocoder.sample_rate == 24000
    assert vocoder.token_mel_ratio == 2


def test_loaded_vocoder_decodes_on_scheduler_thread(tmp_path):
    _write_tiny_artifact(tmp_path)
    vocoder = FunCosyVoice3MlxVocoder.from_pretrained(str(tmp_path))
    stream = mx.new_thread_local_stream(mx.gpu)
    results = []
    errors = []

    def decode():
        try:
            with mx.stream(stream):
                results.append(
                    vocoder.decode(
                        token=[1, 2, 3],
                        prompt_token=[4, 5],
                        prompt_feat=np.zeros((4, 4), dtype=np.float32),
                        embedding=np.ones(3, dtype=np.float32),
                    )
                )
        except Exception as exc:  # pragma: no cover - asserted in parent thread
            errors.append(exc)

    thread = threading.Thread(target=decode, name="scheduler-vocoder-test")
    thread.start()
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert errors == []
    assert len(results) == 1
    assert results[0].shape[0] > 0
    assert np.isfinite(results[0]).all()


def test_loader_accepts_canonical_hift_prefix(tmp_path):
    _write_tiny_artifact(tmp_path, hift_prefix="hift")

    vocoder = FunCosyVoice3MlxVocoder.from_pretrained(str(tmp_path))

    assert vocoder.sample_rate == 24000


def test_loader_validates_explicit_dtype_against_artifact(tmp_path):
    _write_tiny_artifact(tmp_path)

    FunCosyVoice3MlxVocoder.from_pretrained(
        str(tmp_path),
        expected_dtype="float16",
    )
    with pytest.raises(ValueError, match="owned by the converted artifact"):
        FunCosyVoice3MlxVocoder.from_pretrained(
            str(tmp_path),
            expected_dtype="bfloat16",
        )


def test_decode_validates_prompt_alignment(tmp_path):
    _write_tiny_artifact(tmp_path)
    vocoder = FunCosyVoice3MlxVocoder.from_pretrained(str(tmp_path))

    with pytest.raises(ValueError, match="token_mel_ratio"):
        vocoder.decode_mx(
            token=[1],
            prompt_token=[2, 3],
            prompt_feat=np.zeros((3, 4), dtype=np.float32),
            embedding=np.ones(3, dtype=np.float32),
        )


def test_loader_rejects_unconverted_checkpoint(tmp_path):
    (tmp_path / "flow.pt").touch()
    (tmp_path / "hift.pt").touch()

    with pytest.raises(FileNotFoundError, match="converted artifact"):
        FunCosyVoice3MlxVocoder.from_pretrained(str(tmp_path))


def test_loader_rejects_raw_unsanitized_hift_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            "flow.stub": mx.zeros((1,)),
            "hift.conv_pre.weight_g": mx.zeros((1,)),
        },
    )

    with pytest.raises(ValueError, match="raw unsanitized"):
        FunCosyVoice3MlxVocoder.from_pretrained(str(tmp_path))
