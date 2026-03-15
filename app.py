import streamlit as st
import pandas as pd
from google import genai

# -------------------------------------------------------------------
# ページ設定
# -------------------------------------------------------------------
st.set_page_config(page_title="看護略語AI (Gemini版)", page_icon="🏥", layout="wide")

st.title("看護略語 検索AIアシスタント (Gemini版) 🏥")
st.write("スプレッドシート（CSV）をカンニングペーパーにする「一番小さいRAG」のテストアプリです。")

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

df = load_data()
if not df.empty:
    st.sidebar.dataframe(df)
    st.sidebar.info("アプリは、ユーザーが質問したときに、この表の中から関係ある行だけを探し出します。")

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
        
        # 3. 自分のコードで表を検索する（ここでお金をかけずに絞り込む！）
        found_rows = []
        for index, row in df.iterrows():
            abbreviation = str(row['略語']).lower()
            # 質問の中に略語名が含まれていたら、その行を「カンニングペーパー」に追加する
            if abbreviation in user_question.lower():
                found_rows.append(row)
        
        # 検索結果がゼロだった場合
        if len(found_rows) == 0:
            st.warning("スプレッドシートの中に、関係しそうな略語が見つかりませんでした。")
            st.info("【AIへの指示】：すみません、提供された資料の中にはその略語は見つかりませんでした。")
            
        # 検索結果が見つかった場合
        else:
            st.success(f"スプレッドシートから {len(found_rows)} 件の関係しそうなデータを見つけました！")
            
            # 見つけた数件だけのデータを、APIに渡すための文字（テキスト）にまとめる
            context_text = ""
            for row in found_rows:
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

