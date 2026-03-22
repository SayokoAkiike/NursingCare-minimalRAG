import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

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
# ここでAPIキーを入力できるようにする
api_key = st.sidebar.text_input("Gemini APIキーを入力してください（AIza...）", type="password")

st.sidebar.markdown("---")
st.sidebar.header("📄 現在のデータ（カンニングペーパー）")

# 1. データの読み込み
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data.csv")
    except FileNotFoundError:
        st.error("data.csvが見つかりません。")
        return pd.DataFrame()


def row_to_text(row):
    return " ".join(
        [
            str(row.get("略語", "")),
            str(row.get("正式名称", "")),
            str(row.get("意味", "")),
            str(row.get("よく使う場面", "")),
            str(row.get("注意点", "")),
            str(row.get("関連語", "")),
        ]
    )


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
        
        # 3. 無料ローカル埋め込み検索で関係行を絞り込む
        selected_rows = retrieve_related_rows(df, user_question)
        
        # 検索結果がゼロだった場合
        if len(selected_rows) == 0:
            st.warning("スプレッドシートの中に、関係しそうな略語が見つかりませんでした。")
            st.info("【AIへの指示】：すみません、提供された資料の中にはその略語は見つかりませんでした。")
            
        # 検索結果が見つかった場合
        else:
            st.success(f"ベクトル検索で {len(selected_rows)} 件の関係しそうなデータを見つけました！")

            ranking_df = pd.DataFrame(
                [
                    {
                        "略語": df.iloc[idx]["略語"],
                        "類似度": f"{score:.3f}",
                    }
                    for idx, score in selected_rows
                ]
            )
            st.caption("検索上位（類似度）")
            st.dataframe(ranking_df, use_container_width=True, hide_index=True)
            
            # 見つけた数件だけのデータを、APIに渡すための文字（テキスト）にまとめる
            context_text = ""
            for idx, _ in selected_rows:
                row = df.iloc[idx]
                context_text += f"・略語: {row['略語']} (正式名称: {row['正式名称']})\n"
                context_text += f"  意味: {row['意味']}\n"
                context_text += f"  よく使う場面: {row['よく使う場面']}\n"
                context_text += f"  注意点: {row['注意点']}\n\n"
            
            with st.expander("🔍 AIに渡すカンニングペーパーの中身（クリックして確認）"):
                st.code(context_text, language="text")
                st.write("※スプレッドシート全体ではなく、見つけたこの文字だけをAPIに送るから安く安全に済みます！")
            
            # 4. APIに送るデータ（プロンプト）の作成
            prompt = f"""
            あなたは新人看護師を優しくサポートする先輩AIです。
            以下の情報を「カンニングペーパー」として使い、ユーザーの質問に優しく答えてください。
            
            【厳守事項】
            1. 必ずカンニングペーパーに書かれているデータ「のみ」を使って答えてください。
            2. カンニングペーパーに書かれていないこと（一般的な医療知識など）は絶対に答えないでください。
            3. 分からないことや資料にないことは「資料にありません」と素直に答えてください。
            4. 現場で使うときの「注意点」があれば、必ず添えてあげてください。

            【ユーザーの質問】: {user_question}

            【カンニングペーパーの内容】: 
            {context_text}
            """
            
            st.subheader("🤖 AIの回答")
            
            # -------------------------------------------------------------------
            # 5. APIキーが入力されている場合のみ、本物のGeminiを呼び出す
            # -------------------------------------------------------------------
            if api_key.startswith("AIza"):
                try:
                    # 本物のAIと通信中...
                    with st.spinner("Geminiが回答を生成しています..."):
                        # Gemini APIクライアントの準備
                        client = genai.Client(api_key=api_key)
                        
                        # Geminiモデル（gemini-2.5-flashなど）を呼び出す
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt
                        )
                        
                        # 結果の表示
                        st.write(response.text)
                        
                except Exception as e:
                    st.error(f"エラーが発生しました（APIキーが間違っているか、通信エラーです）: {e}")
            else:
                # APIキーがない場合は今まで通りのテストモード（ダミー）
                st.info("💡 画面左側のサイドバーに有効なGemini APIキー（AIza...）が入力されていないため、【テストモード】で動かしています。")
                st.write(f"（APIキーを入力すると、Geminiから以下のような回答が返ってきます）\n\n**テスト回答:** お疲れ様です！お探しの略語については以下の通りです。\n\n{context_text}\n現場で使うときは、特に「注意点」に気をつけてくださいね！")

