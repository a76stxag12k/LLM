```mermaid
flowchart TD

subgraph ローカル LLM 環境
a1[プロンプト入力]-->a2{プロンプト判定}
a2-->|web検索不要|a4[LLM問い合わせ]-->a7
a2-->|web検索必要|a3[web検索クエリ作成]-->a5[web検索]
a5-->a6[LLMで要約]
a6-->a7[結果表示]
end
```
