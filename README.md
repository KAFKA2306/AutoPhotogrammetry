https://kafka2306.github.io/AutoPhotogrammetry/

# AutoPhotogrammetry

実写写真・動画から、監査可能な3D再構成入力、Gaussian Splat PLY、mesh成果物を作るためのrepositoryです。

公開サービスは、3D化前の撮影セット監査です。元画像を削除せず、重複・鮮明度・類似画像・来歴不足を整理し、JSON/HTMLレポートと再構成投入用の選別済み画像セットを生成します。

## 撮影セットを監査する

```bash
python main.py audit --dataset <dataset-id>
```

主な出力:

- `readiness-report.json`
- `readiness-report.html`
- `selected-manifest.json`
- `selected/`

登録画像率、再投影誤差、mesh completeness、実寸精度を測定していない場合は、推測して品質保証しません。

サービス仕様:

- `docs/business/photogrammetry-input-audit.md`
- `docs/business/photogrammetry-readiness-service.md`
- `.github/ISSUE_TEMPLATE/photogrammetry-service.yml`

## Gaussian Splatを生成する

通常の実行入口は `task` です。

```bash
./task run
```

現在のproduction flow:

```text
権利確認済み写真・動画
-> source identity / SHA-256
-> frame extraction / filtering
-> COLMAP camera poses
-> Nerfstudio data conversion
-> Splatfacto
-> Gaussian Splat PLY
-> artifact identity / SHA-256 / size / provenance
```

生成したPLYは通常のGit履歴へ保存しません。artifact storageが利用できない場合はmaterializationを明示的にblockedとし、Git branchやRelease commitを代替保存先にしません。

展示対象sourceの正本は `sources/videos.json` です。下流のWeb / Unity / VRChat表示は `KAFKA2306/vrmine` が担当します。

## 既存datasetで品質を比較する

```bash
./task quality <scene-id> 30000
```

同じprocessed datasetと評価条件で、Splatfactoの候補設定を比較します。GPU実行していない結果を実測PASSとして扱いません。

## 開発

```bash
uv sync
uv run python -m unittest discover -s tests
```

Docker / GPU /外部toolが必要な実行は `task` のpreflightを通します。必要な実行環境が無い場合に別経路へ自動fallbackしません。

## 主要authority

- `AGENTS.md` — 再構成・artifact保存の恒常ルール
- `task` — operator向け実行入口
- `sources/videos.json` — 展示source catalog
- `processing/` — 再構成・評価・artifact処理
- Issue #3 — 撮影セット監査サービスの有償PoC検証
- Issue #33 — Gaussian Splat production catalog
- Issue #110 — Gaussian Splatからmeshへの成果物経路

実測値、artifact identity、個別experimentの結論はREADMEへ重複保存せず、machine-readable artifactまたは該当Issueをauthorityとします。
