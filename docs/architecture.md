# Irodori-TTS-Lite — 内部設計メモ

## 全体像

```
                     +-----------------------------------+
                     |  irodori_tts.inference_runtime    |
                     |  ・InferenceRuntime.from_key      |
                     |  ・_load_checkpoint_for_inference |
                     +----------------+------------------+
                                      ^
                                      |  irodori_tts_lite.patch() で差し込み
                                      |
   +-----------------------+    +-----+---------------------+
   |  weights/             |    |  irodori_tts_lite          |
   |   *.safetensors       |--->|   checkpoint_loader.py     |
   |   (4-bit, metadata    |    |    ・metadata から         |
   |    に量子化レコード)  |    |       quant_layers_json /  |
   +-----------------------+    |       extra_quant_layers_  |
                                |       json を取得          |
                                |    ・DiT block: Fused      |
                                |    ・AdaLN/Encoder: eager  |
                                |   quant_utils.py           |
                                |   fused_int4_linear.py     |
                                +----------------------------+
                                              |
                                              v
                                +----------------------------+
                                | TextToLatentRFDiT (model)  |
                                |  ・blocks.*.attn.{q,k,v,o} |
                                |  ・blocks.*.mlp.{w1,w2,w3} |
                                |  ・blocks.*.adaln.*        |
                                |  ・text_encoder, encoders  |
                                +----------------------------+
```

## 2 段階の量子化フォーマット

### 1. GPTQ uniform 4-bit (DiT block の Linear + AdaLN)

OneCompression が GPTQ で生成する標準フォーマット。`{name}.qweight` `{name}.scales` `{name}.qzeros` `{name}.g_idx` `{name}.bias` を持ち、metadata の `quant_method=autobit` / `quant_layers_json` でレイヤごとの設定（in/out features, wbits, groupsize, actorder）を保持する。

- `qweight`: int32, `(in_features // 8, out_features)` — 4-bit を AutoGPTQ-v1 連続ビット詰めで int32 にパック
- `scales` : fp16, `(in_features // groupsize, out_features)`
- `qzeros` : int32, `(in_features // groupsize, out_features // 8)` — int32 にパック (v1 は `(z + 1) & 0xF` で復元)
- `g_idx`  : int32 — actorder=False では使われないため kernel 側は ignore

### 2. RTN uint8-nibble pack (エンコーダ / cond_module / text_embedding / RMSNorm 拡張)

GPTQ パイプラインに乗らない（小さい・コールドパス）レイヤ用のシンプルな後処理。形式は OneCompression リポの `example/example_dit_encoders_rtn_postpass.py`（再構成中）が出す:

- `_extra.{name}.qweight_u8`: uint8, `(out_features, ceil(in_features, 2) // 2)` — 1 バイトに低位/高位 2 nibble
- `_extra.{name}.scales`    : fp16, `(out_features, num_groups)` — `num_groups == 1` で per-row、`>1` で grouped
- `_extra.{name}.zeros`     : fp16/int, `(out_features, num_groups)`

metadata の `extra_quant_layers_json` でレイヤ一覧と `(in_features, out_features, num_groups)` を保持。

## ロードフロー

`patch()` が `irodori_tts.inference_runtime` の 2 つの関数を差し替える:

1. `_load_checkpoint_for_inference(path)`
   - safetensors の metadata から 2 系統の量子化レコードを取り出し、`_PENDING_SWAPS` / `_PENDING_EXTRA` にバッファリングする
   - 通常の state_dict から `.qweight` などの量子化系キーは抜き取られた状態で上位に返す

2. `InferenceRuntime.from_key(cls, key)`
   - **CPU 上で** モデル骨格を構築（FP32 で ~1.9 GB のため、GPU に置く前に最小化したい）
   - `_PENDING_SWAPS` を走査:
     - 名前が `adaln*` を含むなら、eager-dequant して fp16 `nn.Linear` に
     - 4-bit groupsize=32 / actorder=False / in%32 == 0 / out%8 == 0 を満たすなら `FusedInt4Linear`
     - それ以外は fallback で eager-dequant
   - `_PENDING_EXTRA` を走査:
     - dequant 後の重みでターゲットモジュールの `weight` パラメータを置換（型は変更せず、`nn.Linear` のまま、`RMSNorm` のままなど）
     - shape が一致しないが要素数が一致するケース（RMSNorm の `(heads, head_dim)` が `(in, out)` の形で保存されているケース）は reshape して書き戻し
   - `load_state_dict(strict=False)` で残りの非量子化重み（LayerNorm の bias など）を反映
   - `force_fp16=True` の場合は最後にモデル全体（FusedInt4Linear を除く）を fp16 にキャスト
   - `FusedInt4Linear` がある場合は warmup（Triton JIT のキャッシュを温める）

## なぜ AdaLN は eager-dequant か

- AdaLN は DiT 1 block あたり 12 個の小さな projection（fan_in ~1280, fan_out ~1280）
- 12 block × N RF step（プロダクションは N=6）→ 1 推論で **数百〜数千回** の極小 GEMM
- Triton kernel の launch overhead は ~10 µs / call、cuBLAS fp16 は同等の shape で ~3 µs
- ⇒ Triton で動かすと逆に遅くなる。fp16 LBLAS 化が正解。

## なぜ encoder/cond_module は eager-dequant か

- 1 推論で 1 回だけ呼ばれる（コールドパス）
- weight は 数 MB 〜 数十 MB と小さい
- fp16 化しても **runtime メモリ増加は数十 MB**（ピーク 552 MB に対して誤差レベル）
- 一方、Triton kernel 化すると JIT 起動コスト・autotune コストが推論ごとに乗る
- ⇒ 単純な fp16 nn.Linear / nn.Embedding に展開する方がデプロイが堅い。

## 既知の制限

- `FusedInt4Linear` の fp32 入力 fallback path に shape-check の不整合あり（K-padding が考慮されない）。`force_fp16=True` を推奨。
- カーネルは Ampere 系（compute capability ≥ 8.0）でチューニング済み。Turing 以下は未検証。
- batch=1, seq~250 でのチューニング。大きい batch でカーネル選択ヒューリスティクスが最適から外れる可能性あり。

## 量子化済みウェイトの再現

量子化パイプライン本体（GPTQ uniform 4-bit on DiT blocks / RTN post-pass on encoders / eager-dequant 戦略）の手順書は別リポで公開予定。
