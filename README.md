# AutoPhotogrammetry

[![Test](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml/badge.svg)](https://github.com/KAFKA2306/AutoPhotogrammetry/actions/workflows/test.yml)

**実写動画 → camera poses → Gaussian Splat PLY** を1タスクで再現します。

## Local responsibility

ローカルの責務は **RUNするだけ**です。

```bash
./task run
```

初回だけcloneします。

```bash
git clone https://github.com/KAFKA2306/AutoPhotogrammetry.git
cd AutoPhotogrammetry
./task run
```

環境確認、Docker image build、入力取得、hash検証、3D再構成、training、PLY export、成功判定は `./task run` 側が行います。

実行入口: [`task`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/task)  
GPU環境: [`Dockerfile`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/Dockerfile)  
E2E実装: [`processing/huejotzingo.py`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/processing/huejotzingo.py)

## What runs

```text
[Wikimedia CC0 video]
  -> SHA-256 verification
  -> [FFmpeg] frame extraction
  -> blur / duplicate filtering
  -> [COLMAP] feature_extractor
  -> [COLMAP] sequential_matcher
  -> [COLMAP] mapper
  -> [Nerfstudio] ns-process-data
  -> [Nerfstudio Splatfacto] ns-train splatfacto
  -> [Nerfstudio export] ns-export gaussian-splat
  -> PLY + manifest
```

- FFmpeg: https://ffmpeg.org/
- COLMAP CLI: https://colmap.github.io/cli.html
- Nerfstudio custom data: https://docs.nerf.studio/quickstart/custom_dataset.html
- Nerfstudio Splatfacto: https://docs.nerf.studio/nerfology/methods/splat.html
- Nerfstudio export: https://docs.nerf.studio/reference/cli/ns_export.html
- gsplat: https://github.com/nerfstudio-project/gsplat

## Video candidates

動画候補の正本は [`sources/videos.json`](https://github.com/KAFKA2306/AutoPhotogrammetry/blob/main/sources/videos.json) です。URL、license確認状態、duration、resolution、期待成功度、riskを **20候補**まとめています。

`status: verified` はhashまで固定済み、`status: candidate` は候補段階です。`expected_success` / `score` は実行前のヒューリスティックで、実測結果ではありません。

上位候補:

1. **95 / verified** — [Ex Convento de San Miguel Arcángel, Huejotzingo](https://commons.wikimedia.org/wiki/File:Vista_del_Ex_Convento_de_San_Miguel_Arcángel,_Huejotzingo,_desde_un_dron.webm)
2. **94 / high** — [Museo Nacional del Virreinato + Templo de San Francisco Javier, Tepotzotlán](https://commons.wikimedia.org/wiki/File:Panorámica_del_Museo_Nacional_del_Virreinato_y_Templo_de_San_Francisco_Javier_desde_un_dron.webm)
3. **91 / high** — [Templo de San Marcos](https://commons.wikimedia.org/wiki/File:Fachada_del_Templo_de_San_Marcos_desde_un_dron.webm)
4. **89 / high** — [Calvillo centro](https://commons.wikimedia.org/wiki/File:Calvillo_desde_un_dron_(plaza_principal,_Santa_Cruz,_centro).webm)
5. **88 / high** — [Puente de San Ignacio 03](https://commons.wikimedia.org/wiki/File:Puente_de_San_Ignacio_desde_un_dron_03.webm)

探索元: [Wikimedia Commons — Drone videos from Mexico](https://commons.wikimedia.org/wiki/Category:Drone_videos_from_Mexico)

現在の `./task run` はregistryの `default` である `huejotzingo` を使います。

## Input

固定検証データ:

- source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Vista_del_Ex_Convento_de_San_Miguel_Arcángel,_Huejotzingo,_desde_un_dron.webm)
- author: Luisalvaz
- license: CC0 1.0
- resolution: 1920×1080 transcode
- duration: 232.766 s
- SHA-256: `c9723df1af171d40a5bf1f9530aa3ea881c6f95252ef3f2004f0f1013ab92e30`

確認済みCOLMAP結果:

- registered: **78 / 78 images**
- sparse points: **32,782**
- mean reprojection error: **0.370830 px**

## Output

```text
output/huejotzingo/
├── frames/
├── selected/
├── colmap/
├── nerfstudio-data/
├── runs/
└── manifest.json
```

成功条件は `manifest.json` が次を持つことです。

```json
{
  "status": "success",
  "output": {
    "ply_path": "...",
    "ply_sha256": "...",
    "ply_size_bytes": 123456
  }
}
```

PLY hashが得られるまで成功扱いにしません。

## Scope

このrepoの終点は **Gaussian Splat PLY + provenance** です。

PLYのWeb / Unity / VRChat利用は [`KAFKA2306/vrmine`](https://github.com/KAFKA2306/vrmine) が担当します。
