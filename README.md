# Irodori-TTS-Lite

TTS DiT を 4-bit 量子化したまま推論する、軽量ランタイムです。

オリジナルの FP32 チェックポイント（1.88 GB）を **279 MB のディスク容量** と、DiT 単体で **552 MB のピーク GPU メモリ** で動かせます。音質はほぼ劣化しません。

さらにオプションの `--codec-int4` を付ければ **DACVAE コーデックも 4-bit** で動かせて、エンドツーエンド（DiT + コーデック + トークナイザ）でピーク **約 1 GB**（実測 988.7 MB）になります。

---

## 計測値（実機, RTX PRO 4000 Blackwell, 6 RF step, 単発推論）

### ディスクサイズ

| Mode | Safetensors |
| --- | ---: |
| FP32 | 1888 MB |
| BF16 | 944 MB |
| **INT4 (本リポジトリ)** | **279 MB** |

### GPU メモリ — DiT モデルのみ（DACVAE コーデック・トークナイザ除く）

| Mode | `max_memory_allocated` | `max_memory_reserved` |
| --- | ---: | ---: |
| FP32 | 1916.8 MB | 1992.0 MB |
| FP16 | 978.7 MB | 1004.0 MB |
| **INT4** | **552.3 MB** | **568.0 MB** |

### GPU メモリ — エンドツーエンド推論（`infer.main()` 経由、DACVAE / トークナイザ / activation すべて含む）

| Mode | `--codec-device cpu` | `--codec-device cuda` | `--codec-int4` (cuda) |
| --- | ---: | ---: | ---: |
| FP32 | （未計測） | 2867.7 MB | — |
| BF16 | （未計測） | 1990.6 MB | — |
| **INT4** | **988.7 MB** | 1512.6 MB | **988.7 MB** |

> **DACVAE コーデックも 4-bit にする `--codec-int4` モードを追加しました。**
> 重みを uint8-nibble pack のまま保持して forward の中で 1 レイヤずつ on-the-fly dequant するので、
> Conv 重み 377 → 59 MB（▲ 84%）、エンドツーエンドのピークが **1513 → 989 MB**（▲ 525 MB）になります。
> DiT と DACVAE が同時に走るわけではないので、両者を 4-bit にしても合計は単なる加算にはならず、
> **VRAM 1 GB 弱**でフルパイプラインが回ります。
>
> DACVAE コーデックを GPU に置いて codec も int4 にすれば decode_latent は ~330 ms、
> コーデック fp16 のままなら ~170 ms、CPU に逃がすと ~3.3 s。
> レイテンシ要件が緩い環境では `--codec-device cpu` で更に 500 MB 節約できます。
>
> 再現は [`tools/measure_peak_memory.py`](tools/measure_peak_memory.py) で。

### レイテンシ（参考）

| Mode | sample_rf | decode_latent | total |
| --- | ---: | ---: | ---: |
| FP32 (codec=cuda) | 270 ms | 116 ms | 392 ms |
| BF16 (codec=cuda) | 367 ms | 124 ms | 499 ms |
| INT4 (codec=cuda, fp16) | 296 ms | 172 ms | 473 ms |
| INT4 (codec=cuda, int4) | 292 ms | 333 ms | 629 ms |
| INT4 (codec=cpu)        | 334 ms | 3316 ms | 3.66 s |

---

## 特徴

- **自己完結**: OneCompression を実行時依存に持ちません（量子化済みウェイト + ランタイムだけで動きます）。
- **Triton 高速カーネル**: DiT block の Linear は `FusedInt4Linear`（GPTQ-v1 packed × cuBLAS 級カーネル）で実行。
- **賢い eager-dequant**: AdaLN（ホットパス、起動オーバヘッドが支配的）と エンコーダ系（コールドパス、1 推論あたり 1 回呼び）はロード時に fp16 化して `nn.Linear` に差し替え。
- **既存パイプラインに 1 行で割り込み**: `irodori_tts_lite.patch()` を呼ぶだけで、`irodori_tts.inference_runtime.InferenceRuntime.from_key` が 4-bit セーフテンソルを直接読めるようになります。

---

## クイックスタート

### 1. インストール

```bash
pip install git+https://github.com/kizuna-intelligence/Irodori-TTS-Lite.git
pip install pyopenjtalk            # run_tts.py でテキスト → 秒数の自動推定に使用

# 上流 TTS パイプラインは別途インストールしてください（patch() で実行時にフックします）。
```

重みは初回実行時に [`kizuna-intelligence/Irodori-TTS-Lite-int4`](https://huggingface.co/kizuna-intelligence/Irodori-TTS-Lite-int4) から自動ダウンロードされます（HF キャッシュに保存）。

### 2. 動かす

```bash
# --checkpoint を省略すると HF から自動ダウンロード
python example/run_tts.py \
    --text "こんにちは、メラだよ。テスト中なの！" \
    --output-wav /tmp/sample.wav \
    --no-ref
```

ローカルの safetensors を使いたい場合は `--checkpoint <path>`、別 HF リポを指定したい場合は `--checkpoint hf://<org>/<repo>/<file>` で渡せます。`--no-ref` は話者性を重みに焼き込んだチェックポイント用フラグ。

### 3. メモリ消費を実測する

```bash
# フルパイプラインのピーク (DACVAE 含む)
python tools/measure_peak_memory.py --mode int4 --no-ref --json

# DACVAE を CPU に逃がして VRAM を最小化
python tools/measure_peak_memory.py --mode int4 --no-ref --codec-device cpu --json

# DACVAE もそのまま GPU で int4 化（VRAM 約 1 GB でフル推論）
python tools/measure_peak_memory.py --mode int4 --no-ref --codec-int4 --json
```

未量子化との比較は `--mode bf16` / `--mode fp32` に切り替え + 対応する未量子化 checkpoint を `--checkpoint` で渡してください。

---

## ライブラリとして組み込む

`patch()` で既存の `irodori_tts.inference_runtime` にフックを差し込みます。あとは普段通り `infer.main()` を呼べば 4-bit ファイルがそのまま読み込まれます。

```python
import irodori_tts_lite

irodori_tts_lite.configure(use_fused=True, force_fp16=True)
irodori_tts_lite.patch()

import infer
infer.main()
```

`configure()` で調整できる主なオプション:

| 引数 | デフォルト | 説明 |
| --- | --- | --- |
| `use_fused` | `True` | DiT block の Linear を Triton カーネルで実行する |
| `force_fp16` | `False` | モデル全体を fp16 に強制（カーネルのネイティブ dtype） |
| `disable_eager` | `False` | AdaLN の eager-dequant を無効化（デバッグ用） |
| `codec_int4` | `False` | DACVAE コーデックの NormConv1d / NormConvTranspose1d を int4 packed に置き換える（VRAM ピーク ~525 MB 削減、decode は ~170→330 ms） |
| `codec_int4_groupsize` | `32` | コーデック int4 量子化のグループサイズ |
| `adaln_streaming` | `False` | AdaLN projection を eager-dequant せず、forward 中に都度 dequant する（VRAM ~30-60 MB 削減、推論速度は数 % 悪化見込み）|

> ⚠️ 現状、`FusedInt4Linear` の fp32 入力フォールバックパスに既知の shape-check 制限があります。推論時は `force_fp16=True`（または `example/run_tts.py` のデフォルト）を推奨します。

---

## 仕組み

量子化ウェイトは安全テンソルのメタデータ内に二系統で記録されています:

| スコープ | 形式 | ロード時の置換先 |
| --- | --- | --- |
| DiT block の Linear（q/k/v/o、SwiGLU MLP の w1/w2/w3） | GPTQ uniform 4-bit, groupsize=32 | `FusedInt4Linear`（Triton カーネル） |
| AdaLN projection | GPTQ 4-bit | eager-dequant → fp16 `nn.Linear` |
| エンコーダ / cond_module / text_embedding / RMSNorm 拡張 | RTN 4-bit, uint8-nibble pack | eager-dequant → fp16 module（型はターゲットに合わせる） |

AdaLN を eager-dequant している理由は、12 個の小さな projection × 12 block × N RF step → 数千回の極小 Triton kernel launch になり、起動オーバヘッドが int4 速度メリットを食い潰してしまうため。エンコーダ系は逆に「呼び出し回数が少なく、サイズも小さい」ためカーネルを使う旨味がほぼ無いので fp16 にしてしまった方が単純です。

詳細な設計判断は [`docs/architecture.md`](docs/architecture.md) を参照してください。

---

## ディレクトリ構成

```
.
├── README.md                 # 本ファイル
├── LICENSE                   # MIT
├── pyproject.toml            # パッケージ定義
├── irodori_tts_lite/         # ランタイム本体（自己完結、外部依存は torch/triton/safetensors のみ）
│   ├── __init__.py
│   ├── checkpoint_loader.py  # inference_runtime へのモンキーパッチ
│   ├── fused_int4_linear.py  # Triton カーネル（OneCompression からベンダリング）
│   ├── quant_utils.py        # GPTQ-v1 unpack + RTN-extras dequant
│   ├── packed_conv.py        # DACVAE 用 int4 packed Conv1d / ConvTranspose1d
│   └── weights.py            # HF からの自動ダウンロード
├── example/
│   └── run_tts.py            # 単発合成サンプル
├── tools/
│   ├── measure_peak_memory.py    # フルパイプラインのピーク VRAM 計測
│   └── quantize_dacvae_poc.py    # DACVAE int4 PoC（ベースラインと SNR 比較）
└── docs/
    └── architecture.md       # 内部設計
```

---

## ハードウェア要件

- CUDA 対応 GPU（compute capability 8.0 以上を推奨。Triton カーネルが Ampere 系チューニング）
- VRAM:
  - DiT 単体なら 1 GB あれば十分（実測ピーク 552 MB + CUDA コンテキスト分）
  - DACVAE コーデックを GPU に置く場合は 2 GB 推奨（実測ピーク 1513 MB）

CPU 推論は未対応です（DACVAE のみ CPU offload 可）。

---

## 開発

```bash
pip install -e .[infer]
```

ベンチや追加の動作確認用スクリプトは `tools/` 配下に追加していく方針です。

---

## 上流

- TTS パイプライン本体: [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
- DACVAE コーデック: [Aratako/Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)

どちらも Aratako 氏の成果物を 4-bit 量子化して使わせていただいています。

---

## ライセンス

[MIT License](LICENSE)

`irodori_tts_lite/fused_int4_linear.py` は OneCompression プロジェクト（Fujitsu Ltd. 著作）からのベンダリングです。元のライセンス表記を保持しています。

---

Copyright © 2025-2026 Kizuna Intelligence contributors.
（`irodori_tts_lite/fused_int4_linear.py` のみ Copyright © 2025-2026 Fujitsu Ltd.）
