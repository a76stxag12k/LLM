from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from concurrent.futures import ThreadPoolExecutor
import inspect
import streamlit as st
import requests
import json

API_SERVER_URL = "http://localhost:11434/api/chat"

# ルータープロンプトの定義
router_prompt = PromptTemplate(
    template="""  
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    あなたはユーザーの質問を適切にルーティングする専門家です。
    Web検索が必要か、不要化の判断をして下さい。
    Web検索が必要な場合を 3 、不要な場合を 0 として
    必要性を 3 段階の 整数だけで答えて下さい。
    質問: {question}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question"],
)

# クエリ変換のプロンプト定義
query_prompt = PromptTemplate(
    template="""  
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    ユーザーの質問を、最適なWeb検索クエリに変換してください。
    クエリは英語のみを使用し、記号を含まない形式で作成してください。
    
    質問: {question}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question"],
)

# web検索の結果を要約するプロンプト定義
summary_prompt = PromptTemplate(
    template="""  
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    あなたは文章を適切に要約する専門家です。
    下記の文章を読んで、質問に対し要約し回答して下さい。
    
    質問: {question}

    文章: {content}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "content"],
)

# グラフの状態を定義
class GraphState(TypedDict):
    original_question: str  # オリジナルの質問
    question: str           # 処理中の質問
    generation: str         # LLMまたは検索の結果
    search_query: str       # 生成された検索クエリ
    context: str            # 文脈
    action: str             # 実行するアクション (web検索またはllm)

# 質問に基づいて、Web検索を実行すべきか、直接回答をLLMで生成すべきかを判断する関数
def route_question(state: GraphState):
    """
    質問に基づいて、Web検索を行うべきか、LLMを使用して回答を生成するべきかを判断する。
    """
    # プロンプトに検索ワードを適用
    state['original_question'] = state["question"]
    user_query = state["question"]
    formatted_prompt = router_prompt.format(question=state['question'])
    state["question"] = formatted_prompt

    # LLM 問い合わせ
    state = inquire_llm(state)

    print('確度:[' + state["generation"] + ']')

    # LLMの出力が 確度:2 より大きい場合、web検索が必要
    if int(state["generation"]) > 2:
        print('web 検索...')
        state["action"] = "web"         # web検索
    else:
        print('llm 問い合わせ...')
        state["action"] = "llm"         # 回答生成
    state["question"] = user_query

    return state

# 条件付きエッジの実装
def should_search(state: GraphState):
    """
    Web検索が必要かどうかを決定する。
    """
    if "web" == state["action"]:
        return "web"
    else:
        return "llm"

# ユーザーの質問をWeb検索用のクエリに変換する関数
def transform_query(state: GraphState):
    """
    ユーザーの質問をWeb検索用のクエリに変換する。
    """
    # LLM に問い合わせるためのプロンプトを作成
    formatted_prompt = query_prompt.format(question=state['question'], )
    state["question"] = formatted_prompt
    return state

# ユーザーの質問を元にWeb検索を実行し、その結果を state に保存する関数
def web_search(state: GraphState):
    """
    ユーザーの質問に基づきWeb検索を実行し、その結果をcontextに保存する。
    """
    # 必要な入力データの取得
    question = state["question"]
    context = state["context"]

    # LLM 問い合わせ
    state = inquire_llm(state)
    state["context"] = state["generation"]

    user_query = ''
    test_response = ''

    try:
        # Web検索ツールの定義
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=25)
        web_search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

        # Web検索ツールの実行
        user_query = state["generation"]
        test_response = web_search_tool.invoke(user_query)

        print('[search from web:' + user_query + ']')
        print('[result from web:' + test_response + ']')
    except Exception as e:
        # エラー生成
        state["generation"] = f"Web Search Error: {e}"
        print(state["generation"])
        return state

    # LLM に問い合わせるためのプロンプトを作成
    formatted_prompt = summary_prompt.format(question=state['original_question'], content=test_response)
    state["question"] = formatted_prompt

    # LLM 問い合わせ
    state = inquire_llm(state)
    state["context"] = state["generation"]

    return state

# LLM の回答を生成する関数
def generate(state: GraphState):
    """
    LLM を用いて回答を生成する。
    """
    # 必要な入力データの取得
    question = state["question"]
    context = state["context"]

    # LLM 問い合わせ
    state = inquire_llm(state)
    state["context"] = state["generation"]
    return state

# LLM に問い合わせする関数
def inquire_llm(state: GraphState):
    """
    LLM に質問を投げ、生成された回答を受け取る。
    """
    # 必要な入力データの取得
    question = state["question"]
    context = state["context"]

    print('[To llm:' + question + ']')

    # Ollama 用にメッセージを構築
    headerstr = {"Content-Type": "application/json"}
    jsonstr = {
        "model": "yourmodel",
        "messages": [{
            "role": "user",
            "content": question,
        }]
    }

    state["generation"] = ''

    # LLM の呼び出し
    response = requests.post(API_SERVER_URL, headers=headerstr, json=jsonstr)

    # レスポンスが成功した場合
    if response.status_code == 200:

        try:
            # 複数行のJSON文字列をリストの形式に変換
            json_lines = response.text.strip().splitlines()

            # 応答を逐次的に表示
            for json_line in json_lines:

                contents = json.loads(json_line)
                if "message" in contents:
                    result = contents["message"].get("content", "") # 結果からテキストを取り出す
                    if state["generation"] is None:
                        state["generation"] = ''
                    state["generation"] += result

        except KeyError as e:
            # エラー生成
            state["generation"] = f"KeyError: Missing expected key in the response: {e}"
        except requests.exceptions.JSONDecodeError as e:
            # エラー生成
            state["generation"] = f"JSON Decode Error: {e}"
        except Exception as e:
            # エラー生成
            state["generation"] = f"Error: {e}"
    else:
        state["generation"] = f"Error: {response.status_code}"

    print('[result from llm:' + state["generation"] + ']')

    return state

# ワークフローの定義
workflow = StateGraph(GraphState)

# ノード追加
workflow.add_node("route_question", route_question)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)

# 条件付きエッジ
workflow.add_conditional_edges(
    "route_question",
    should_search,
    {
        "web": "transform_query",
        "llm": "generate"
    }
)
workflow.add_edge("transform_query", "web_search")          # 変換から検索
workflow.add_edge("web_search", "generate")                 # 検索から生成
workflow.add_edge("generate", END)                          # 生成から終了

# 初期ノードを設定
workflow.set_entry_point("route_question")

# コンパイルしてエージェントを作成
local_agent = workflow.compile()

state = GraphState()

# エージェントを実行する関数
def run_agent(query):
    """
    与えられたクエリをエージェントで処理し、生成された結果を返す。
    """
    output = local_agent.invoke({"question": query})
    return output["generation"]

# Streamlit UI
st.title("ローカル LLM の動作確認")

user_query = st.text_input("動作確認の内容を入力して下さい。:", "")

if st.button("クエリを実行"):
    if user_query:
        st.write(run_agent(user_query))
