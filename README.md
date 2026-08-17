# AutoPhotogrammetry — input → processing → output

[![Test](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml/badge.svg)](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml)

許諾を確認できる実写画像・動画を `input/` に置き、`processing/` のコードだけで処理し、frame・camera pose・Nerfstudio dataset・Gaussian Splat PLY・manifestを `output/` に生成するリポジトリです。

独自のSfM、3D Gaussian Splatting training、rasterizerは実装しません。FFmpeg、COLMAP、Nerfstudio Splatfactoを外部CLIとして利用し、入力hash、tool version、command、return code、stdout/stderr、生成物hashを追跡します。

## 構造

```text
AutoPhotogrammetry/
├── input/                  # 原本。Gitには大容量データをcommitしない
├── processing/             # 処理コードだけ
│   ├── collection.py
│   ├── image_selection.py
│   ├── video.py
│   ├── photogrammetry.py
│   ├── nerfstudio.py
│   └── provenance.py
├── output/                 # 中間生成物と最終成果物。Gitにはcommitしない
├── tests/
├── main.py                 # input/outputを指定する薄いCLI
└── requirements.txt
```

原則は一方向です。

```text
external source
  -> input/
  -> processing/
  -> output/
```

`processing/` に入力データや生成物を置きません。`output/` を手修正して正本にしません。

## 現在の実データ

実測対象はメキシコ・Huejotzingoの **Ex Convento de San Miguel Arcángel** です。

Wikimedia Commons 1080p transcode:

https://upload.wikimedia.org/wikipedia/commons/transcoded/3/34/Vista_del_Ex_Convento_de_San_Miguel_Arc%C3%A1ngel%2C_Huejotzingo%2C_desde_un_dron.webm/Vista_del_Ex_Convento_de_San_Miguel_Arc%C3%A1ngel%2C_Huejotzingo%2C_desde_un_dron.webm.1080p.vp9.webm

確認済み原本:

- container: WebM
- video codec: VP9
- resolution: 1920 × 1080
- duration: 232.766 s
- size: 115,502,605 bytes
- SHA-256: `c9723df1af171d40a5bf1f9530aa3ea881c6f95252ef3f2004f0f1013ab92e30`

確認済みCOLMAP結果:

- input images: 78
- registered images: 78
- registration rate: 100%
- submodels: 1
- sparse points: 32,782
- mean reprojection error: 0.370830 px

実GPUでのSplatfacto trainingとGaussian Splat PLY exportはまだ未完了です。CPU CIのmock testをGPU training成功とは扱いません。

## 画像を入力する

明示したHTML pageから取得する場合:

```bash
python -m pip install -r requirements.txt

python main.py collect \
  --dataset example \
  --page-url 'https://example.org/licensed-photo-page' \
  --keyword building
```

取得物は `input/example/images/` に保存されます。検索engineをscrapeしません。

選別:

```bash
python main.py select --dataset example
```

選別結果は `output/example/selected/` にcopyされます。原本は削除しません。

## Huejotzingo動画から再現する

Ubuntu 24.04でrepository rootから実行します。

### 1. 依存を入れる

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
sudo apt-get update
sudo apt-get install -y ffmpeg colmap
```

### 2. 原本を `input/` に取得する

```bash
mkdir -p input/huejotzingo

curl --fail --location --retry 3 \
  --user-agent 'AutoPhotogrammetry/0.3 (https://github.com/KAFKA2306/AutoPhotogrammetry)' \
  --output input/huejotzingo/source.webm \
  'https://upload.wikimedia.org/wikipedia/commons/transcoded/3/34/Vista_del_Ex_Convento_de_San_Miguel_Arc%C3%A1ngel%2C_Huejotzingo%2C_desde_un_dron.webm/Vista_del_Ex_Convento_de_San_Miguel_Arc%C3%A1ngel%2C_Huejotzingo%2C_desde_un_dron.webm.1080p.vp9.webm'

echo 'c9723df1af171d40a5bf1f9530aa3ea881c6f95252ef3f2004f0f1013ab92e30  input/huejotzingo/source.webm' | sha256sum --check
```

### 3. frameを `output/` に生成する

実測条件は3秒に1枚、幅1024 pxです。

```bash
mkdir -p output/huejotzingo/frames

ffmpeg -hide_banner -loglevel error \
  -i input/huejotzingo/source.webm \
  -vf 'fps=1/3,scale=1024:-2' \
  -q:v 2 \
  output/huejotzingo/frames/frame-%06d.jpg
```

この入力では78 frameが期待値です。

### 4. blur / near-duplicateを除く

```bash
mkdir -p output/huejotzingo/selected

python -c "from pathlib import Path; import json; from processing.image_selection import select_video_frames; frames=sorted(Path('output/huejotzingo/frames').glob('frame-*.jpg')); print(json.dumps(select_video_frames(frames, 'output/huejotzingo/selected'), indent=2))"
```

Huejotzingoの3秒間隔入力では78枚すべてをCOLMAPへ渡した実績があります。

### 5. COLMAP camera poseを `output/` に生成する

```bash
rm -rf output/huejotzingo/colmap
mkdir -p output/huejotzingo/colmap/sparse

colmap feature_extractor \
  --database_path output/huejotzingo/colmap/database.db \
  --image_path output/huejotzingo/selected \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 0 \
  --SiftExtraction.max_image_size 1024 \
  --SiftExtraction.max_num_features 4096

colmap sequential_matcher \
  --database_path output/huejotzingo/colmap/database.db \
  --SiftMatching.use_gpu 0

colmap mapper \
  --database_path output/huejotzingo/colmap/database.db \
  --image_path output/huejotzingo/selected \
  --output_path output/huejotzingo/colmap/sparse

colmap model_analyzer --path output/huejotzingo/colmap/sparse/0
```

### 6. 既存COLMAP modelをNerfstudio datasetへ変換する

Nerfstudioは別途GPU環境へinstallし、`ns-process-data`、`ns-train`、`ns-export`をPATHから実行できる状態にします。

現在のNerfstudioでは `--skip-colmap` と `--colmap-model-path` があり、既存COLMAP modelを再利用できます。`colmap-model-path` はNerfstudio output directoryからの相対pathです。

```bash
ns-process-data images \
  --data output/huejotzingo/selected \
  --output-dir output/huejotzingo/nerfstudio-data \
  --skip-colmap \
  --colmap-model-path ../colmap/sparse/0
```

### 7. Splatfacto trainingとPLY export

```bash
python -c "import json; from processing.nerfstudio import run_splatfacto_export; result=run_splatfacto_export('output/huejotzingo/nerfstudio-data', 'output/huejotzingo/runs'); print(json.dumps(result, indent=2))"
```

成功時のrun directoryには、少なくとも次が残ります。

- `manifest.json`
- training / export stdout・stderr
- Nerfstudio `config.yml`
- checkpoint path
- exportされた `.ply`
- 入力画像SHA-256
- Nerfstudio / gsplat version
- command、timestamps、return code
- PLY size、SHA-256

training/export失敗は成功として扱いません。

## 外部photogrammetry backend

`processing/photogrammetry.py` はMeshroom、VisualSFM、COLMAPの外部実行を担当します。

設定も同じmoduleに置き、独立した `config.py` は持ちません。

環境変数:

- `AUTOPHOTOGRAMMETRY_MESHROOM_EXECUTABLE`
- `AUTOPHOTOGRAMMETRY_VISUALSFM_EXECUTABLE`
- `AUTOPHOTOGRAMMETRY_COLMAP_EXECUTABLE`

外部softwareが存在しない場合は自動installやfallbackを行わず停止します。`shell=True`は使いません。

## テスト

```bash
python -m unittest discover -s tests -v
```

CIはGPU trainingを実行しません。検証対象は、画像選別、動画command、provenance、photogrammetry runner、Nerfstudio command/manifest/failure pathです。

## repository間の責務

`KAFKA2306/AutoPhotogrammetry`:

```text
real images / video
  -> provenance
  -> frame selection
  -> camera poses
  -> Nerfstudio Splatfacto
  -> Gaussian Splat PLY
```

`KAFKA2306/vrmine`:

```text
Gaussian Splat PLY + provenance
  -> Web / Unity / VRChat compatibility validation
```

viewerやVRChat galleryはこのrepositoryに置きません。

## 制約

- robots.txt、利用規約、著作権licenseを自動判定しない
- 検索engineをscrapeしない
- 生成AIの別角度画像をSfM / 3DGS入力へ混ぜない
- 独自SfM / 3DGS training / rasterizerを実装しない
- source video、checkpoint、大容量PLYをGitへcommitしない
- COLMAP登録率だけでGaussian Splat品質を保証しない
- Huejotzingoで実GPU PLYを生成するまではE2E成功と記載しない

## 一次資料

- Nerfstudio custom data: https://docs.nerf.studio/quickstart/custom_dataset.html
- Nerfstudio Splatfacto: https://docs.nerf.studio/nerfology/methods/splat.html
- Nerfstudio `ns-export`: https://docs.nerf.studio/reference/cli/ns_export.html
- Nerfstudio COLMAP converter source: https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/process_data/colmap_converter_to_nerfstudio_dataset.py
- COLMAP CLI: https://colmap.github.io/cli.html
