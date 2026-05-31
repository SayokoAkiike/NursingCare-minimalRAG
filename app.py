import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

CSV_COLUMNS = ["略語", "正式名称", "意味", "よく使う場面", "注意点", "関連語"]

# -------------------------------------------------------------------
# ページ設定
# -------------------------------------------------------------------
st.set_page_config(page_title="看護略語AI (Gemini版)", page_icon="🏥", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700&family=Shippori+Mincho:wght@600&display=swap');

    :root {
        --bg1: #f4fbf8;
        --bg2: #e6f3ff;
        --ink: #102a43;
        --muted: #486581;
        --accent: #0f9d8a;
        --accent-2: #2f80ed;
        --card: rgba(255, 255, 255, 0.72);
        --line: rgba(15, 157, 138, 0.2);
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 20%, rgba(47, 128, 237, 0.17), transparent 32%),
            radial-gradient(circle at 84% 24%, rgba(15, 157, 138, 0.15), transparent 35%),
            linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 100%);
        color: var(--ink);
    }

    html, body, [class*="css"] {
        font-family: 'Zen Kaku Gothic New', sans-serif;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0.02em;
    }

    .hero {
        background: linear-gradient(120deg, rgba(16, 42, 67, 0.95), rgba(15, 157, 138, 0.9));
        color: #f8fcff;
        border-radius: 18px;
        padding: 24px 26px;
        margin: 8px 0 20px;
        box-shadow: 0 18px 38px rgba(16, 42, 67, 0.25);
    }

    .hero h1 {
        color: #ffffff;
        margin: 0 0 8px 0;
        font-family: 'Shippori Mincho', serif;
        font-weight: 600;
    }

    .hero p {
        margin: 0;
        color: #dff5ee;
    }

    .stat-card {
        background: var(--card);
        border: 1px solid var(--line);
        backdrop-filter: blur(8px);
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 10px 26px rgba(16, 42, 67, 0.08);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(245,252,249,0.94));
        border-right: 1px solid rgba(16, 42, 67, 0.07);
    }

    .stButton button {
        border-radius: 999px;
        border: 1px solid rgba(15, 157, 138, 0.4);
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: #fff;
        font-weight: 700;
        padding: 0.45rem 1.2rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(47, 128, 237, 0.24);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>看護略語 検索AIアシスタント</h1>
        <p>無料ローカル埋め込み検索で、必要な略語だけを先に抽出してからAIに質問できます。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# サイドバー：設定とデータ表示
# -------------------------------------------------------------------
st.sidebar.header("⚙️ 設定")
# APIキーは st.secrets を優先し、未設定時にサイドバー入力にフォールバックする
secret_api_key = st.secrets.get("GEMINI_API_KEY", None)
api_key_input = st.sidebar.text_input("Gemini APIキーを入力してください（AIza...）", type="password")
api_key = secret_api_key or api_key_input

st.sidebar.markdown("---")
st.sidebar.header("📄 現在のデータ（コンテキスト）")

# 1. データの読み込み
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv", encoding="utf-8")
        missing = [c for c in CSV_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"data.csv に必要な列がありません: {missing}")
            return pd.DataFrame()
        return df
    except FileNotFoundError:
        st.error("data.csvが見つかりません。")
        return pd.DataFrame()


def row_to_text(row):
    return " ".join(str(row.get(col, "")) for col in CSV_COLUMNS)


@st.cache_resource
def build_embedding_index(df_source):
    texts = [row_to_text(row) for _, row in df_source.iterrows()]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve_related_rows(df_source, question, top_k=3, score_threshold=0.08):
    vectorizer, matrix = build_embedding_index(df_source)
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, matrix).flatten()

    # 類似度が高い順に並べて、閾値以上のみ採用
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    selected = [(idx, score) for idx, score in ranked if score >= score_threshold][:top_k]
    return selected

df = load_data()
if not df.empty:
    st.sidebar.dataframe(df)
    st.sidebar.info("質問文と各行をベクトル化し、コサイン類似度で関係の深い行を先に取り出します。")

stats_col1, stats_col2, stats_col3 = st.columns(3)
with stats_col1:
    st.markdown(f"<div class='stat-card'><b>登録略語</b><br>{len(df)} 件</div>", unsafe_allow_html=True)
with stats_col2:
    st.markdown("<div class='stat-card'><b>検索方式</b><br>無料ローカル埋め込み</div>", unsafe_allow_html=True)
with stats_col3:
    st.markdown("<div class='stat-card'><b>回答モデル</b><br>Gemini 2.5 Flash</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# メイン画面：ユーザーの質問を受け取る
# -------------------------------------------------------------------
st.subheader("AIに質問してみましょう")
user_question = st.text_input("分からない略語を入力してください（例：「SOBって何？」「BPの意味は？」）")

if st.button("検索してAIに聞く"):
    if not user_question:
        st.warning("質問を入力してください。")
    elif df.empty:
        st.error("データが読み込めていません。")
    else:
        st.write("---")

        # 3. ローカル埋め込み検索で関係行を絞り込む
        selected_rows = retrieve_related_rows(df, user_question)

        # 検索結果がゼロだった場合
        if len(selected_rows) == 0:
            st.warning("辞書の中に、関係しそうな略語が見つかりませんでした。")
            st.info("提供された資料の中にはその略語は見つかりませんでした。")

        # 検索結果が見つかった場合
        else:
            st.success(f"ベクトル検索で {len(selected_rows)} 件の関連データを見つけました。")

            abbr_col, full_col, meaning_col, usage_col, caution_col, related_col = CSV_COLUMNS
            ranking_df = pd.DataFrame(
                [
                    {
                        "略語": df.iloc[idx][abbr_col],
                        "類似度": f"{score:.3f}",
                    }
                    for idx, score in selected_rows
                ]
            )
            st.caption("検索上位（類似度）")
            st.dataframe(ranking_df, use_container_width=True, hide_index=True)

            # 見つかった行をコンテキストとしてまとめる
            context_text = ""
            for idx, _ in selected_rows:
                row = df.iloc[idx]
                context_text += f"・略語: {row[abbr_col]} (正式名称: {row[full_col]})\n"
                context_text += f"  意味: {row[meaning_col]}\n"
                context_text += f"  よく使う場面: {row[usage_col]}\n"
                context_text += f"  注意点: {row[caution_col]}\n\n"

            with st.expander("🔍 AIに渡すコンテキストの中身（クリックして確認）"):
                st.code(context_text, language="text")
                st.write("※辞書全体ではなく、類似度上位の行のみをコンテキストとしてAPIに送信しています。")

            # 4. APIに送るプロンプトの作成
            prompt = f"""
            あなたは新人看護師を優しくサポートする先輩AIです。
            以下の情報をコンテキストとして使い、ユーザーの質問に優しく答えてください。

            【厳守事項】
            1. 必ずコンテキストに書かれているデータ「のみ」を使って答えてください。
            2. コンテキストに書かれていないこと（一般的な医療知識など）は絶対に答えないでください。
            3. 分からないことや資料にないことは「資料にありません」と素直に答えてください。
            4. 現場で使うときの「注意点」があれば、必ず添えてあげてください。

            【ユーザーの質問】: {user_question}

            【コンテキストの内容】:
            {context_text}
            """

            st.subheader("🤖 AIの回答")

            # -------------------------------------------------------------------
            # 5. APIキーが有効な場合にのみ Gemini を呼び出す
            # -------------------------------------------------------------------
            valid_api_key = api_key and api_key.startswith("AIza")
            if valid_api_key:
                try:
                    # Gemini APIクライアントを初期化してリクエストを実行する
                    with st.spinner("Geminiが回答を生成しています..."):
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt
                        )
                        st.write(response.text)
                except Exception as e:
                    st.error(f"エラーが発生しました（APIキーが間違っているか、通信エラーです）: {e}")
            else:
                if api_key:
                    st.error("入力されたGemini APIキーが無効です。AIzaで始まる有効なキーを入力してください。")
                else:
                    st.error("Gemini APIキーが未設定です。サイドバーか st.secrets に有効なキーを設定してください。")
                st.info("💡 有効なキーがないため、テストモードで動作しています。")
                st.write(f"（APIキーを入力すると、Geminiから以下のような回答が返ってきます）\n\n**テスト回答:** お疲れ様です！お探しの略語については以下の通りです。\n\n{context_text}\n現場で使うときは、特に「注意点」に気をつけてくださいね！")
