# AutoPhotogrammetry — 実写から監査可能な3D再構成へ

[![Test](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml/badge.svg)](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml)

許諾を確認できる実写画像・動画を入力として、出典とSHA-256を保存し、FFmpeg、COLMAP、Nerfstudio Splatfactoを外部CLIとして実行するリポジトリです。

独自のSfM、3D Gaussian Splatting training、rasterizerは実装しません。実行したcommand、tool version、return code、stdout/stderr、入力hash、生成物hashを保存します。

## 現在の実データ

対象はメキシコ・Huejotzingoの **Ex Convento de San Miguel Arcángel** です。

使用したWikimedia Commons 1080p transcode:

https://upload.wikimedia.org/wikipedia/commons/transcoded/3/34/Vista_del_Ex_Convento_de_San_Miguel_Arc%C3%A1ngel%2C_Huejotzingo%2C_desde_un_dron.webm/Vista_del_Ex_Convento_de_San_Miguel_Arc%C3%A1ngel%2C_Huejotzingo%2C_desde_un_dron.webm.1080p.vp9.webm

実際に取得したファイル:

- container: WebM
- video codec: VP9
- resolution: 1920 × 1080
- duration: 232.766 s
- size: 115,502,605 bytes
- SHA-256: `c9723df1af171d40a5bf1f9530aa3ea881c6f95252ef3f2004f0f1013ab92e30`

COLMAP実測結果:

- input images: 78
- registered images: 78
- registration rate: 100%
- submodels: 1
- sparse points: 32,782
- mean reprojection error: 0.370830 px

実GPUでのSplatfacto trainingとGaussian Splat PLY exportはまだ未実行です。CPU CIのmock testをGPU training成功とは扱いません。

## Huejotzingoを再現する

Ubuntu 24.04で、repository rootから次を実行します。

### 1. Python依存を入れる

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
sudo apt-get update
sudo apt-get install -y ffmpeg colmap
```

### 2. 動画を取得する

```bash
mkdir -p work/huejotzingo
curl --fail --location --retry 3 \
  --user-agent 'AutoPhotogrammetry/0.1 (https://github.com/KAFKA2306/AutoPhotogrammetry)' \
  --output work/huejotzingo/source.webm \
  'https://upload.wikimedia.org/wikipedia/commons/transcoded/3/34/Vista_del_Ex_Convento_de_San_Miguel_Arc%C3%A1ngel%2C_Huejotzingo%2C_desde_un_dron.webm/Vista_del_Ex_Convento_de_San_Miguel_Arc%C3%A1ngel%2C_Huejotzingo%2C_desde_un_dron.webm.1080p.vp9.webm'
```

取得後にhashを確認します。

```bash
echo 'c9723df1af171d40a5bf1f9530aa3ea881c6f95252ef3f2004f0f1013ab92e30  work/huejotzingo/source.webm' | sha256sum --check
```

動画metadataを確認します。

```bash
ffprobe -v error \
  -show_entries format=duration,size,format_name:stream=codec_name,width,height \
  -of json \
  work/huejotzingo/source.webm
```

### 3. 3秒間隔で78 frameを抽出する

COLMAP実測時は3秒に1枚へ落とし、幅1024 pxへ縮小しました。

```bash
mkdir -p work/huejotzingo/frames
ffmpeg -hide_banner -loglevel error \
  -i work/huejotzingo/source.webm \
  -vf 'fps=1/3,scale=1024:-2' \
  -q:v 2 \
  work/huejotzingo/frames/frame-%06d.jpg
```

枚数を確認します。

```bash
find work/huejotzingo/frames -maxdepth 1 -name 'frame-*.jpg' | wc -l
```

この入力では `78` が期待値です。

### 4. repositoryのframe選別を実行する

```bash
mkdir -p work/huejotzingo/selected
python -c "from pathlib import Path; from video_pipeline import select_video_frames; import json; frames=sorted(Path('work/huejotzingo/frames').glob('frame-*.jpg')); print(json.dumps(select_video_frames(frames, 'work/huejotzingo/selected'), indent=2))"
```

Huejotzingoの3秒間隔入力では78枚すべてをCOLMAPへ渡した実績があります。

### 5. COLMAPでcamera poseを推定する

```bash
rm -f work/huejotzingo/colmap.db
rm -rf work/huejotzingo/sparse
mkdir -p work/huejotzingo/sparse

colmap feature_extractor \
  --database_path work/huejotzingo/colmap.db \
  --image_path work/huejotzingo/selected \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 0 \
  --SiftExtraction.max_image_size 1024 \
  --SiftExtraction.max_num_features 4096

colmap sequential_matcher \
  --database_path work/huejotzingo/colmap.db \
  --SiftMatching.use_gpu 0

colmap mapper \
  --database_path work/huejotzingo/colmap.db \
  --image_path work/huejotzingo/selected \
  --output_path work/huejotzingo/sparse
```

生成されたmodelを解析します。

```bash
colmap model_analyzer --path work/huejotzingo/sparse/0
```

2026-08-18時点のGitHub Actions実測値は次です。

```text
Cameras: 1
Images: 78
Registered images: 78
Points: 32782
Mean reprojection error: 0.370830px
```

### 6. Nerfstudioを入れる

このrepositoryの`requirements.txt`にはNerfstudioを固定していません。GPU環境でNerfstudioを別途installし、`ns-train`と`ns-export`がPATHから実行できる状態にします。

確認command:

```bash
ns-train --help
ns-export --help
python -c "from importlib.metadata import version; print('nerfstudio', version('nerfstudio')); print('gsplat', version('gsplat'))"
```

### 7. Nerfstudio datasetを作る

既存COLMAP modelを再利用する場合は、Nerfstudioの`ns-process-data`で`work/huejotzingo/selected`と`work/huejotzingo/sparse/0`を使ってdatasetを作成します。

Nerfstudioのinstalled versionで引数名を確認してから実行してください。

```bash
ns-process-data images --help
```

このstepは、repository内でまだHuejotzingo実GPU E2Eとして固定できていません。COLMAPを再計算せず既存modelを使うことを優先します。

### 8. Splatfacto trainingとPLY exportを実行する

Nerfstudio datasetを`work/huejotzingo/nerfstudio-data`へ作成した後、repositoryのrunnerを使います。

```bash
python -c "from video_pipeline import run_splatfacto_export; import json; result=run_splatfacto_export('work/huejotzingo/nerfstudio-data', 'work/huejotzingo/runs'); print(json.dumps(result, indent=2))"
```

`run_splatfacto_export()`は内部で実在する`ns-train splatfacto`と`ns-export gaussian-splat`を実行します。成功した場合のみ`status: success`を返します。

run directoryには次を保存します。

- `manifest.json`
- `train.stdout.log`
- `train.stderr.log`
- `export.stdout.log`
- `export.stderr.log`
- Nerfstudioが生成した`config.yml`
- Nerfstudio checkpointへのpath
- exportされたPLY

`manifest.json`には次を保存します。

- 全入力画像のpath、size、SHA-256
- Nerfstudio version
- gsplat version
- 実行したtraining command
- 実行したexport command
- start timestamp
- end timestamp
- return code
- config path
- checkpoint path
- PLY path
- PLY size
- PLY SHA-256
- 失敗したphase

## 画像Webページから収集する場合

`main.py`は利用者が明示したHTML pageから画像を取得します。以下は動作確認用の架空URLではなく、利用者自身が利用許諾を確認した実在URLへ置き換えて実行してください。READMEでは未確認URLを実例として掲載しません。

CLI引数は次で確認できます。

```bash
python main.py --help
```

`main.py`が保存する主要情報は取得元page URL、image URL、MIME、画像寸法、SHA-256です。取得画像を削除せず、選別結果は別directoryへcopyします。

## 外部フォトグラメトリbackend

`photogrammetry.py`が対応するbackendは次の3つです。

- Meshroom
- VisualSFM
- COLMAP

実行ファイルは`BackendConfig(executable=...)`、JSON設定、または次の環境変数で指定します。

- `AUTOPHOTOGRAMMETRY_MESHROOM_EXECUTABLE`
- `AUTOPHOTOGRAMMETRY_VISUALSFM_EXECUTABLE`
- `AUTOPHOTOGRAMMETRY_COLMAP_EXECUTABLE`

外部softwareが存在しない場合は自動installやfallbackを行わず停止します。`subprocess`実行では`shell=True`を使いません。

## テスト

```bash
python -m unittest discover -s tests -v
```

2026-08-18時点で通常CIは16 testsを実行しています。

検証内容:

- 異なる解像度でも特徴量長が一定
- 異なる解像度同士でSSIMを計算可能
- 選別時に元画像を削除しない
- 空入力を処理可能
- 空白を含むpathを1引数として扱う
- 外部実行ファイル欠落時にfail-closed
- backendごとのmanifest / stdout / stderr分離
- `ns-train splatfacto` command construction
- `ns-export gaussian-splat` command construction
- training failure時にfailed manifestを保存
- contract testでcheckpoint / PLY metadataとSHA-256を保存

通常CIではGPU trainingを実行しません。Huejotzingo COLMAP実測用の一時workflowもmainから削除済みです。

## repository間の責務

`KAFKA2306/AutoPhotogrammetry`は実写入力、provenance、frame選別、camera pose、Splatfacto training、Gaussian Splat PLY生成を担当します。

`KAFKA2306/vrmine`は生成済みGaussian Splat PLYをWeb、Unity、VRChat側で表示・互換性検証する側です。

同じtraining pipelineを2 repositoryへ実装しません。

## 利用条件と限界

- robots.txt、利用規約、著作権licenseを自動判定しません
- 利用者が明示したHTML pageだけを画像収集対象にします
- 検索engineの無断scraping機能はありません
- 同一対象、十分なviewpoint overlap、照明条件を自動証明しません
- cluster番号は3D形状やcamera poseを意味しません
- SSIMは画像類似度であり、reprojection errorや3D精度ではありません
- COLMAP 100% registrationだけではGaussian Splat品質を保証しません
- Huejotzingoで実GPU PLYが生成されるまではE2E成功と記載しません

**README最終監査:** 2026-08-18
