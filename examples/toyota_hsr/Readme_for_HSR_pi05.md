# Readme_for_HSR_pi05

`/home/tell/Devenv/ICLR_Competition/airoa-moma` のような AIRoA MoMa 形式データで、`lerobot/pi05_base` を学習開始するための手順です。  
この README では、`examples/toyota_hsr/train_pi05_on_airoa_moma.sh` の使い方と、設定できる変数を説明します。

## 0. 前提

1. `conda activate vla`
2. このリポジトリ直下にいること
   - `/home/tell/Devenv/ICLR_Competition/ICRAtam@home_lerobot`
3. `pi0.5` 依存が入っていること
   - 例: `pip install -e ".[pi]"`

## 1. なぜ `prepare` が必要か

AIRoA MoMa には `action.absolute` / `action.relative` はありますが、`action` キーがありません。  
LeRobot の学習パイプラインは `action` を参照するため、学習前に `action` エイリアスを追加したコピーを作るのが安全です。

この処理を `examples/toyota_hsr/prepare_airoa_moma_for_pi05.py` が行います。

## 2. 最短実行（最初のバッチ確認）

```bash
conda activate vla
cd /home/tell/Devenv/ICLR_Competition/ICRAtam@home_lerobot

bash examples/toyota_hsr/train_pi05_on_airoa_moma.sh
```

デフォルトでは以下を実行します。

- inspect スクリプトでキー確認
- `action.relative` を `action` にコピーした学習用データを作成
- `steps=1` で学習起動（最初のバッチ確認用）

## 3. 実運用向け実行例

```bash
SRC_DATASET_ROOT=/home/tell/Devenv/ICLR_Competition/airoa-moma \
PREPARED_DATASET_ROOT=/home/tell/Devenv/ICLR_Competition/airoa-moma-pi05 \
TRAIN_DATASET_ROOT=/home/tell/Devenv/ICLR_Competition/airoa-moma-pi05 \
DATASET_REPO_ID=airoa-moma-hsr-pi05-local \
DATASET_EPISODES='[0,1,2,3,4,5,6,7,8,9]' \
ACTION_KEY=action.relative \
POLICY_PATH=lerobot/pi05_base \
POLICY_DEVICE=cuda \
DTYPE=bfloat16 \
BATCH_SIZE=4 \
NUM_WORKERS=8 \
STEPS=3000 \
LOG_FREQ=10 \
OUTPUT_DIR=outputs/train/toyota_hsr_pi05_run1 \
bash examples/toyota_hsr/train_pi05_on_airoa_moma.sh
```

## 4. 変数一覧

| 変数名 | 既定値 | 取りうる値 / 意味 |
|---|---|---|
| `PYTHON_BIN` | `python` | 実行に使う Python。例: `python`, `/path/to/python` |
| `SRC_DATASET_ROOT` | `/home/tell/Devenv/ICLR_Competition/airoa-moma` | 元データセット root（`meta/info.json` を含む） |
| `PREPARED_DATASET_ROOT` | `/home/tell/Devenv/ICLR_Competition/airoa-moma-pi05` | `action` エイリアス追加済みデータの出力先 |
| `TRAIN_DATASET_ROOT` | `$PREPARED_DATASET_ROOT` | 学習で使うデータ root。準備済みデータを直接指定可能 |
| `DATASET_REPO_ID` | `airoa-moma-hsr-pi05-local` | LeRobot 内部識別子。ローカル用途なら任意文字列で可 |
| `DATASET_EPISODES` | `[0]` | 学習対象エピソード。JSON 形式リスト文字列（例: `'[0,1,2]'`） |
| `ACTION_KEY` | `action.relative` | `action` へコピーする元キー。`action.relative`(11次元) 推奨、`action.absolute`(8次元) も可 |
| `RUN_INSPECT` | `true` | `true/false`。学習前に `inspect_airoa_moma_features.py` を実行するか |
| `SKIP_PREPARE` | `false` | `true/false`。既に準備済みデータを使う場合は `true` |
| `FORCE_PREPARE` | `false` | `true/false`。`PREPARED_DATASET_ROOT` が既存でも削除して再生成 |
| `SYMLINK_VIDEOS` | `true` | `true/false`。動画をコピーせずシンボリックリンク化して高速化 |
| `POLICY_PATH` | `lerobot/pi05_base` | 学習元の重み/設定。Hub ID かローカル checkpoint ディレクトリ |
| `POLICY_DEVICE` | `cuda` | `cuda` / `cpu` |
| `DTYPE` | `float32` | `float32` / `bfloat16` |
| `COMPILE_MODEL` | `false` | `true/false`。`torch.compile` 利用 |
| `GRADIENT_CHECKPOINTING` | `false` | `true/false`。メモリ節約（計算は遅くなる） |
| `FREEZE_VISION_ENCODER` | `false` | `true/false`。視覚エンコーダ凍結 |
| `TRAIN_EXPERT_ONLY` | `false` | `true/false`。行動 expert 側のみ学習 |
| `NORMALIZATION_MAPPING` | `{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}` | 正規化方式。例: `{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}` |
| `BATCH_SIZE` | `1` | 学習バッチサイズ |
| `NUM_WORKERS` | `2` | dataloader ワーカー数 |
| `STEPS` | `1` | 学習ステップ数 |
| `LOG_FREQ` | `1` | ログ出力間隔 |
| `OUTPUT_DIR` | `outputs/train/toyota_hsr_pi05` | 出力先ディレクトリ（存在済みだとエラー） |
| `INSPECT_DATASET_REPO_ID` | `airoa-moma-local` | inspect 実行時に使う dataset repo id |

## 5. よくあるエラーと対処

- `KeyError: 'action'`
  - `SKIP_PREPARE=true` で未加工データを指定している可能性があります。`SKIP_PREPARE=false` で再実行してください。
- `All image features are missing from the batch`
  - `TRAIN_DATASET_ROOT/meta/info.json` の画像キーと実データが壊れていないか確認してください。
- `Output directory ... already exists`
  - `OUTPUT_DIR` を変更してください。

## 6. 関連ファイル

- 学習スクリプト: `examples/toyota_hsr/train_pi05_on_airoa_moma.sh`
- データ準備: `examples/toyota_hsr/prepare_airoa_moma_for_pi05.py`
- キー確認: `scripts/inspect_airoa_moma_features.py`
