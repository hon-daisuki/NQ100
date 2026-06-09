# NQ100

NQ100 / USTEC の1時間足データを使って、特徴量作成、教師ラベル作成、市場状態AI、売買ルールのバックテストを進めるための作業リポジトリです。

## 方針

- GitHubにはコード、ノートブック、設定サンプルだけを置きます。
- Google DriveにはCSV、学習済みモデル、バックテスト結果などの大きなファイルを置きます。
- ColabではGoogle Driveをマウントして、Drive上のデータを直接読み込みます。

## 推奨フォルダ

```text
NQ100/
  notebooks/        Colabで開くノートブック
  src/              再利用するPythonコード
  configs/          Driveパスなどの設定サンプル
  reports/          小さなレポート置き場
```

## Google Drive側の想定

```text
/content/drive/MyDrive/CFD機械学習/backtest_ready/
  USTEC_features_all.csv
```

実際のファイル名やフォルダ名が違う場合は、`configs/drive_paths.example.json` またはノートブック先頭の `DATA_DIR` と `CSV_NAME` を変更してください。

## Colabで開く

GitHubにpush後、以下のURL形式でColabから開けます。

```text
https://colab.research.google.com/github/hon-daisuki/NQ100/blob/main/notebooks/16_backtest_colab.ipynb
```

## 最初の実行

1. Colabで `notebooks/16_backtest_colab.ipynb` を開く
2. `drive.mount("/content/drive")` を実行する
3. `DATA_DIR` と `CSV_NAME` をDrive上の実データに合わせる
4. データ確認、ラベル作成、学習、バックテストを順番に実行する

