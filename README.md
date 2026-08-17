# AutoPhotogrammetry — 実写から監査可能な3D再構成へ

[![Test](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml/badge.svg)](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml)

許諾を確認できる実写画像・動画を入力として、**出典とハッシュを保持したまま画像選別、COLMAPによるcamera pose推定、Nerfstudio SplatfactoによるGaussian Splat学習・PLY exportへ接続する**ためのリポジトリです。

独自のSfM、3D Gaussian Splatting training、rasterizerは実装しません。既存toolを外部CLIとして使い、入力・version・command・return code・生成物hashを追跡できることを優先します。

## 現在の到達点

実装済み:

- 明示したWebページからの実写画像取得
- URL、MIME、寸法、SHA-256、取得元を`manifest.json`へ保存
- 固定長特徴量によるクラスタリングと非破壊画像選別
- 動画の`ffprobe`情報取得とsource provenance記録
- FFmpeg用frame抽出command生成
- scene-change候補抽出
- blur / near-duplicate frame選別
- Meshroom / VisualSFM / COLMAPの外部実行
- Nerfstudio `ns-train splatfacto` / `ns-export gaussian-splat` の外部実行runner
- Nerfstudio / gsplat version、入力画像SHA-256、command、timestamps、return code、logs、checkpoint、PLY hash/sizeのmanifest化
- training / export失敗時のfail-closed処理

実データで確認済み:

- Huejotzingoの連続ドローン動画から78 frameをCOLMAPへ入力
- registered images: **78 / 78**
- registration rate: **100%**
- submodel: **1**
- sparse points: **32,782**
- mean reprojection error: **0.370830 px**

未完了:

- 実GPUでのHuejotzingo Splatfacto training
- 実データからのGaussian Splat `.ply` export
- source video SHA-256から最終PLY SHA-256までのE2E lineage確認

したがって、現時点では**「実データからGaussian Splat PLYを生成済み」とは主張しません**。CPU CIのmock testは外部CLI契約とmanifest生成を検証するもので、GPU training成功の代替ではありません。

## 正準パイプライン

```text
licensed real photos / single-take video
  -> provenance + source SHA-256
  -> frame extraction / filtering
  -> COLMAP camera registration
  -> Nerfstudio dataset
  -> ns-train splatfacto
  -> ns-export gaussian-splat
  -> splat.ply + SHA-256
```

生成AIで作った別角度画像はSfM / 3DGS入力へ混ぜません。見た目が自然でも、同じ実在対象の形状・模様・camera geometryが視点間で一致する保証がないためです。

## 画像収集

```bash
python -m pip install -r requirements.txt
python main.py \
  --page-url "https://example.org/licensed-photo-page" \
  --keyword building \
  --work-dir work
```

複数ページ・キーワードは引数を繰り返せます。

```bash
python main.py \
  --page-url "https://example.org/page-a" \
  --page-url "https://example.org/page-b" \
  --keyword building \
  --keyword architecture
```

主な出力:

```text
work/
├── collected/
│   ├── <sha256>.jpg
│   └── manifest.json
├── clusters.json
└── selected/
    └── <sha256>.jpg
```

## 動画処理

`video_pipeline.py`には以下を置いています。

- `probe_video()` — ffprobe metadata取得
- `scene_cut_times()` — scene-change候補時刻の抽出
- `extract_frames_command()` — FFmpeg frame抽出command生成
- `frame_timestamp_records()` — frameと元動画timestampの対応付け
- `select_video_frames()` — blur / near-duplicate除去
- `write_source_manifest()` — source page、media URL、author、license、SHA-256等の保存

動画・抽出frame・checkpoint・PLYのような大容量生成物はGit履歴へcommitしません。

## 外部フォトグラメトリ

`photogrammetry.py`はMeshroom、VisualSFM、COLMAPの実行commandを引数配列で構築し、`shell=True`を使用しません。外部softwareを自動インストールせず、実行ファイルが見つからない場合は具体的な設定方法を示して停止します。

実行ファイルは`BackendConfig(executable=...)`、JSON設定、または次の環境変数で指定します。

- `AUTOPHOTOGRAMMETRY_MESHROOM_EXECUTABLE`
- `AUTOPHOTOGRAMMETRY_VISUALSFM_EXECUTABLE`
- `AUTOPHOTOGRAMMETRY_COLMAP_EXECUTABLE`

各実行は`<output_root>/<backend>/<run_id>/`へ分離し、`manifest.json`、`stdout.log`、`stderr.log`、生成物一覧を保存します。

## Nerfstudio Splatfacto

Nerfstudioとgsplatはこのrepositoryへvendorせず、実行環境へ別途installします。

Pythonからの実行入口:

```python
from video_pipeline import run_splatfacto_export

result = run_splatfacto_export(
    "path/to/nerfstudio-data",
    "runs",
)
print(result["manifest_path"])
```

runnerは`ns-train`と`ns-export`が存在しない場合に停止します。成功時は1 runごとに少なくとも次を保存します。

```text
runs/splatfacto/<run-id>/
├── manifest.json
├── train.stdout.log
├── train.stderr.log
├── export.stdout.log
├── export.stderr.log
└── export/
    └── *.ply
```

manifestには以下を記録します。

- input image count
- per-image SHA-256
- Nerfstudio version
- gsplat version
- training / export command
- start / end timestamp
- return code
- config path
- checkpoint path
- PLY path
- PLY size
- PLY SHA-256
- failed phase

## テスト

```bash
python -m unittest discover -s tests -v
```

現在の通常CIはCPUだけで実行し、以下を検証します。

- 異なる解像度でも特徴量長が一定
- 異なる解像度同士のSSIM
- 選別が元画像を削除しない
- 空入力の安全な処理
- 空白を含むpathを1引数として扱う
- 外部実行ファイル欠落時のfail-closed
- backendごとのrun / manifest / log分離
- Splatfacto training / export command construction
- training失敗時のmanifest
- 成功contract時のcheckpoint / PLY metadataとSHA-256

通常CIではGPU trainingを実行しません。実データCOLMAP検証のために使用した一時workflowもroutine CIから削除済みです。

## 責務境界

`AutoPhotogrammetry`:

```text
real images / video
  -> provenance
  -> reconstruction / training
  -> Gaussian Splat PLY + evidence
```

`vrmine`:

```text
Gaussian Splat PLY + evidence
  -> Web / Unity / VRChat側での表示・互換性検証
```

training pipelineを両repositoryへ二重実装しません。

## 利用条件と限界

- robots.txt、利用規約、著作権licenseを自動判定しません
- 利用者が明示したHTML pageだけを画像収集対象にします
- 検索engineの無断scraping機能はありません
- 同じ対象物、十分なviewpoint overlap、照明条件を自動証明しません
- cluster番号は3D形状やcamera poseを意味しません
- 外部backendのinstall、license、GPU要件、入力互換性は利用者が確認します
- SSIMは画像類似度であり、reprojection errorや3D精度の代替ではありません
- COLMAP registration成功だけではGaussian Splat品質を保証しません
- GPU実機でPLYを生成するまではE2E 3DGS成功とは扱いません

以前のREADMEにあった「最高品質」「再構成精度90%以上」等の再現不能な表現は使用しません。

**README最終監査:** 2026-08-18
