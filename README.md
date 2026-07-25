# NQ100

NQ100 / USTEC の東京時間トレード戦略を検証するためのリポジトリです。

## 目的

東京時間に数十分から数時間だけ保有し、小さい利幅でも勝率の高い候補を探します。
現在は、1分足データから以下のようなトレード方法を教師ラベル化して探索します。

- 東京時間 8時から12時台にエントリー
- take profit / stop loss / 最大保有時間を複数パターンで探索
- 2024年以前を学習、2025年を検証、2026年をテストとして時系列分割
- 学習、検証、テストすべてで勝率80%以上かつ平均ポイントがプラスの候補だけを採用

## Webアプリ

`docs/` 配下にGitHub Pages向けの静的Webアプリがあります。

```text
docs/index.html
docs/tokyo_scalp_results.json
docs/tokyo_scalp_models.csv
```

GitHub Pagesを有効にする場合は、GitHubのリポジトリ設定で Pages の Source を `main` ブランチの `/docs` にしてください。

## 戦略探索の実行

ローカルでGoogle Driveが `G:` にマウントされている場合:

```powershell
python scripts/run_tokyo_scalp_search.py --years 2023 2024 2025 2026 --output-dir docs
```

Colabで実行する場合は、Google Driveをマウントしたうえで以下のように実行できます。

```python
!python scripts/run_tokyo_scalp_search.py --years 2023 2024 2025 2026 --output-dir docs
```

## 既存Colabノートブック

バックテスト用ノートブック:

```text
https://colab.research.google.com/github/hon-daisuki/NQ100/blob/main/notebooks/16_backtest_colab.ipynb
```

## 注意

勝率80%以上の候補は、過去データに対する探索結果です。
特に小さい利確幅と大きい損切り幅の組み合わせは、勝率が高くても1回の負けが重くなる可能性があります。
実運用前にはスプレッド、約定、滑り、取引停止時間、ロット管理を含めてデモ口座で検証してください。
