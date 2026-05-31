# 看護略語検索 AI アシスタント

看護現場で使われる略語を、ローカル埋め込み検索（TF-IDF）で候補を絞り込み、
Gemini API で回答を生成する RAG 構成の Streamlit アプリです。

## 技術構成

| レイヤー | 使用技術 |
|------|------|
| フロントエンド | Streamlit |
| 埋め込み検索 | scikit-learn（TF-IDF + コサイン類似度） |
| 回答生成 | Google Gemini 2.5 Flash |
| データ | CSV（看護略語辞書） |

## RAG の処理フロー

1. ユーザーの質問を TF-IDF でベクトル化
2. 略語辞書全行とのコサイン類似度を計算
3. スコア上位 3 件のみをプロンプトに注入
4. Gemini が辞書データの範囲内で回答を生成

辞書外の情報は回答しないようプロンプトで制約しています。

## セットアップ

```bash
pip install -r requirements.txt
streamlit run app.py
```

Gemini API キーは Streamlit の secrets 管理または
サイドバーから入力してください（`AIza...` 形式）。

## データ形式

`data.csv` に以下の列が必要です。

| 列名 | 内容 |
|------|------|
| 略語 | 例: SOB |
| 正式名称 | 例: Shortness of Breath |
| 意味 | 日本語説明 |
| よく使う場面 | 臨床での使用文脈 |
| 注意点 | 誤用・混同しやすい点 |
| 関連語 | 類似略語 |

## 既知の制限

- 検索精度は TF-IDF の特性上、表記ゆれに弱い（将来的に sentence-transformers への移行を検討）
- APIキーはサイドバー入力のため、共有環境では `st.secrets` の使用を推奨
