# Toyota HSR x pi0_base (AIRoA MoMa)

このディレクトリは、`lerobot/pi0_base` を AIRoA MoMa (Toyota HSR) に合わせて学習開始するための再現手順です。

前提:

- `pi0` 実行に必要な `transformers` 依存が入っていること
- `lerobot/pi0_base` にアクセス可能であること

## 1. データセット取得（ゲート付きデータ向け）

AIRoA MoMa はゲート付きの場合があるため、**ローカル保存済みデータを前提**にします。

前提ディレクトリ例:

- `DATASET_ROOT=/path/to/airoa-moma`
- `DATASET_ROOT/meta/info.json` が存在
- `DATASET_ROOT/data/` と `DATASET_ROOT/meta/` が存在

## 2. 先にキーを検査（必須）

学習前に実データのキー名と shape を確認してください。

```bash
python scripts/inspect_airoa_moma_features.py \
  --dataset_root "$DATASET_ROOT" \
  --dataset_repo_id "$DATASET_REPO_ID"
```

出力で必ず確認する項目:

- `observation.*` キー一覧
- `action` キー
- 画像キー候補
- `observation.state` の shape
- `action` の shape
- 1サンプルの dtype と shape

## 3. observation_rename_map 例

inspect 結果で確定した実キーを、pi0 側の想定キーへ変換します。

例:

- `observation.images.head_camera_rgb -> observation.images.right_wrist_0_rgb`
- `observation.images.hand_camera_rgb -> observation.images.left_wrist_0_rgb`

`observation.images.base_0_rgb` は渡しません。欠損視点として扱います。

## 4. state_action_32_adapter 設定例

`pi0_base` の 32 次元 state/action に合わせるため、adapter を有効化します。

### index_embedding 例

- `mode=index_embedding`
- `raw_state_dim=8`
- `raw_action_dim=11`
- `state_index_map=[0,1,2,3,4,6,11,12]`
- `action_index_map=[0,1,2,3,4,6,11,12,13,14,15]`

### linear_projection 例

- `mode=linear_projection`
- `projection_init=random_orthonormal_columns`
- `projection_seed=0`
- 後処理で 11 次元へ戻す場合: `decode_action_to_raw=true`

## 5. 学習起動例（最初のバッチ確認まで）

`index_embedding` で開始する例:

```bash
bash examples/toyota_hsr/train_pi0_base_on_airoa_moma.sh
```

起動ログで、少なくとも以下を確認してください。

- 学習ループが開始される
- 最初の batch が前処理を通過して `step` が進む

## 6. linear_projection で学習する場合

`train_pi0_base_on_airoa_moma.sh` の以下を切り替えます。

- `--policy.state_action_32_adapter.mode=linear_projection`
- 必要なら `--policy.state_action_32_adapter.decode_action_to_raw=true`
