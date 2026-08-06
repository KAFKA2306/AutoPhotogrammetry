# AutoPhotogrammetry — 実写画像の収集・選別プロトタイプ

明示されたWebページから、許諾を確認した実写画像を収集し、出典・ハッシュを保存した上で、固定長特徴量によるクラスタリングと非破壊選別を行う研究プロトタイプです。

## 重要な設計判断

**生成AIで作った別角度画像は、SfM/MVS入力へ混ぜません。** 見た目が自然でも、同じ実在対象の3D形状・模様・カメラ幾何が視点間で一致する保証がないためです。現パイプラインは実写だけを対象とし、出力に`generated_views_used: false`を記録します。

## 監査で修正した問題

- 一つの`main.py`へ疑似モジュールを連結した後、存在しないモジュールをimportしていた構造を単一CLIへ修正
- HTTPタイムアウト、ステータス、Content-Type、最大容量、画像デコード検査を追加
- URL内の埋込み認証情報を拒否
- SHA-256で重複を除去し、取得元ページ・画像URL・寸法・MIMEを`manifest.json`へ保存
- 解像度依存のLBP全画素展開を、固定長LBPヒストグラムへ変更
- HOG・色ヒストグラムを固定解像度で計算
- DBSCAN前に特徴量を標準化
- 鮮明度を平均LaplacianではなくLaplacian分散へ変更
- 異なる解像度の画像を同一サイズへ正規化してSSIMを計算
- 選別時の`rename`を廃止し、元画像を保持したままコピー
- Stable Diffusion・CUDA依存と未使用依存を削除
- 回帰テストを追加

## 実行

```bash
python -m pip install -r requirements.txt
python main.py \
  --page-url "https://example.org/licensed-photo-page" \
  --keyword building \
  --work-dir work
```

複数ページ・キーワードは引数を繰り返します。

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

## 外部フォトグラメトリ実行

`photogrammetry.py` はMeshroom、VisualSFM、COLMAPの実行コマンドを引数配列で構築し、`shell=True`を使用しません。アプリケーションから外部ソフトを自動インストールせず、実行ファイルが見つからない場合は具体的な設定方法を示して停止します。

実行ファイルは、`BackendConfig(executable=...)`、JSON設定、または次の環境変数で指定します。

- `AUTOPHOTOGRAMMETRY_MESHROOM_EXECUTABLE`
- `AUTOPHOTOGRAMMETRY_VISUALSFM_EXECUTABLE`
- `AUTOPHOTOGRAMMETRY_COLMAP_EXECUTABLE`

設定例:

```json
{
  "meshroom": {
    "executable": "C:/Tools/Meshroom/meshroom_photogrammetry.exe",
    "extra_args": []
  },
  "colmap": {
    "executable": "/usr/bin/colmap",
    "extra_args": ["--quality", "medium"]
  }
}
```

各実行は`<output_root>/<backend>/<run_id>/`へ分離され、`manifest.json`、`stdout.log`、`stderr.log`、生成物一覧を保存します。対応OSは各外部ソフトが提供するWindowsまたはLinux環境です。VisualSFMのコマンドライン仕様は配布版によって異なるため、実環境で追加引数を確認してください。

## テスト

```bash
python -m unittest discover -s tests -v
```

検証対象:

- 元画像の解像度が異なっても特徴量長が一定
- 異なる解像度同士でもSSIMを計算できる
- 選別が元ファイルを削除しない
- 空入力を安全に処理する
- 空白を含むパスを1引数として安全に扱う
- 外部実行ファイルがない場合に自動インストールせず停止する
- backendごとにrun、manifest、ログ、成果物を分離する

## 利用条件と限界

- 本ツールはrobots.txt・利用規約・著作権ライセンスを自動判定しません
- 利用者が明示したHTMLページだけを取得対象にします
- 検索エンジンの無断スクレイピング機能はありません
- 同じ対象物・同じ撮影条件・十分な視点重複を自動証明しません
- クラスタ番号は3D形状やカメラ姿勢を意味しません
- 外部バックエンドの導入、ライセンス、GPU要件、入力互換性は利用者が確認します
- SSIMは生成レンダーと参照画像の画像類似度であり、再投影誤差やメッシュ精度の代替ではありません
- 再投影誤差、登録画像率、メッシュ完全性を測るまでは、フォトグラメトリー品質を主張できません

以前のREADMEにあった「最高品質」「再構成精度90%以上」等は、再現可能な証拠がないため削除しています。

**README最終監査:** 2026-08-06
