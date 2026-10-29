# Document Assistant プロジェクト手順書

Document Assistant プロジェクトへようこそ！このプロジェクトでは、LangChain と LangGraph を使って高度なドキュメント処理システムを構築します。金融・医療関連のドキュメントに対して、質問への回答、要約、計算を行える AI アシスタントを作成します。

## プロジェクト概要

このドキュメントアシスタントは、LangGraph によるマルチエージェント・アーキテクチャを採用し、ユーザーのさまざまな種類のリクエストに対応します。

- **Q&A エージェント**: ドキュメントの内容に関する具体的な質問に回答する
- **要約エージェント**: ドキュメントの要約を作成し、重要なポイントを抽出する
- **計算エージェント**: ドキュメント内のデータに対して数値計算を行う

### 前提条件

- Python 3.9 以上
- OpenAI API キー

### インストール

1. リポジトリをクローンする:

```bash
cd <repository_path>
```

2. 仮想環境を作成する:

```bash
python -m venv venv
source venv/bin/activate  # Windows の場合: venv\Scripts\activate
```

3. 依存パッケージをインストールする:

```bash
pip install -r requirements.txt
```

4. `.env` ファイルを作成する:

```bash
cp .env.example .env
# .env を編集して OpenAI API キーを追加する
```

### アシスタントの実行

```bash
python main.py
```

## プロジェクト構成

```
doc_assistant_project/
├── src/
│   ├── schemas.py        # Pydantic モデル
│   ├── retrieval.py      # ドキュメント検索
│   ├── tools.py          # エージェント用ツール
│   ├── prompts.py        # プロンプトテンプレート
│   ├── agent.py          # LangGraph ワークフロー
│   └── assistant.py      # メインエージェント
├── sessions/             # 保存された会話セッション
├── main.py               # エントリーポイント
├── requirements.txt      # 依存パッケージ
└── README.md             # このファイル
```

## エージェント・アーキテクチャ

LangGraph エージェントは以下のワークフローに従います。

![](./docs/langgraph_agent_architecture.png)

## 実装タスク

### 1. スキーマの実装 (schemas.py)

#### タスク 1.1: AnswerResponse スキーマ

構造化された Q&A 応答のための Pydantic モデルを、以下のフィールドで作成してください。

- `question`: ユーザーの元の質問 (string)
- `answer`: 生成された回答 (string)
- `sources`: 参照したソースドキュメント ID のリスト (list of strings)
- `confidence`: 0〜1 の信頼度スコア (float)
- `timestamp`: 応答が生成された日時 (datetime)

**目的**: このスキーマにより回答の形式が統一され、どのドキュメントが参照されたかを追跡できます。

#### タスク 1.2: UserIntent スキーマ

意図 (intent) 分類のための Pydantic モデルを、以下のフィールドで作成してください。

- `intent_type`: 分類された意図 ("qa"、"summarization"、"calculation"、または "unknown")
- `confidence`: 分類の信頼度 (0〜1 の float)
- `reasoning`: 分類の根拠の説明 (string)

**目的**: このスキーマにより、システムはユーザーがどの種類のリクエストをしているかを理解し、適切なエージェントにルーティングできます。

### 2. エージェント状態の実装 (agent.py, assistant.py)

#### タスク 2.1: AgentState のプロパティ

`AgentState` クラスはすでに定義されていますが、その構造を理解しておく必要があります。

- `user_input`: 現在のユーザー入力
- `messages`: LangGraph のメッセージアノテーション付きの会話メッセージ
- `intent`: 分類されたユーザーの意図
- `next_step`: グラフ内で次に実行するノード
- `conversation_summary`: 直近の会話の要約
- `active_documents`: 現在話題になっているドキュメント ID
- `current_response`: 構築中の応答
- `tools_used`: 現在のターンで使用されたツールのリスト
- `session_id` と `user_id`: セッション管理用
- `actions_taken`: 実行されたエージェントノードのリスト (タスク 2.6 で追加)

#### タスク 2.2: 意図分類関数

`classify_intent` 関数を実装してください。
`classify_intent` 関数はグラフの最初のノードです。その目的は、ユーザーの入力と(存在すれば)メッセージ履歴の両方を LLM に渡して問い合わせ、意図を分類させることで、グラフがリクエストを適切なノードに振り分けられるようにすることです。
この関数のコードの一部はすでに用意されていますが、以下の手順で完成させる必要があります。

1. `llm` を `UserIntent` スキーマで構造化出力を使うように設定する
2. `prompts.py` の `get_intent_classification_prompt()` 関数を呼び出してプロンプトを作成する(ヒント: 返された値に対して `format` を呼び出し、`user_input` と `conversation_history` を渡す必要があります)
3. `prompts.py` のプロンプトを読み、どの入力変数が必要か、また LLM に何を返すよう求めているかを必ず理解する
4. プロンプトを使って LLM を呼び出す
5. 分類された `intent` に基づいて `next_step` を設定する条件分岐を実装する:
   - "qa" --> "qa_agent"
   - "summarization" --> "summarization_agent"
   - "calculation" --> "calculation_agent"
   - デフォルト --> "qa_agent"
6. `actions_taken = ["classify_intent"]` で状態を更新し、新しい `intent` の値と `next_step` も含めたうえで、更新された状態を返す

**重要な概念**:

- 構造化された応答には `llm.with_structured_output(UserIntent)` を使用する
- この関数は `actions_taken`、`intent`、`next_step` を含む状態の更新を返す必要がある

#### タスク 2.3: 計算エージェントと要約エージェントの完成

`agent.py` の `qa_agent` ノードのコードを確認してください。受け取るパラメータ、プロンプトの取得と構築方法、構造化出力の強制方法、状態オブジェクトの更新方法に注目してください。

1. `calculation_agent` と `summarization_agent` 関数を同じパターンで実装する。ただし、一部の関数呼び出しはそれぞれのノード固有の値を受け取るよう変更する必要がある
2. プロンプトテンプレートを正しく取得すること
3. 各ノードに対応する構造化出力スキーマの定義を確認し、適切な関数に渡す
4. `qa_agent` ノードが更新するのと同じフィールドをすべて含んだ、更新済みの状態オブジェクトを返すこと

#### タスク 2.4: Update Memory 関数の完成

以下の手順で `update_memory` 関数を完成させてください。

1. config パラメータから llm を取り出す(ヒント: そのために関数を変更する必要があるかもしれません)
2. 構造化出力を強制するために正しいスキーマを渡す
3. conversation_summary、active_documents、next_step で状態を更新する

**目的**: この関数は会話のコンテキストを維持し、ターンをまたいでドキュメントの参照を追跡します。

#### タスク 2.5: ワークフローの作成

以下を行う `create_workflow` 関数を完成させてください。

1. すべてのエージェントノード (classify_intent、qa_agent、summarization_agent、calculation_agent、update_memory) を追加する
2. `add_conditional_edges` メソッドで、各意図タイプを対応するエージェントノードにマッピングする
3. 各エージェントから update_memory へのエッジを追加する
4. 戻り値を修正し、チェックポインター付きでコンパイルされるようにする(タスク 2.6 参照)

**グラフ構造**:

```
classify_intent --> [qa_agent|summarization_agent|calculation_agent] --> update_memory --> END
```

#### タスク 2.6: 状態とメモリの永続化

状態リデューサーと永続メモリの使い方を練習するため、`AgentState` とワークフローを以下のように拡張してください。

1. `AgentState` スキーマの `actions_taken` フィールドに `operator.add` リデューサーを追加する。これにより、ターン中に実行された各エージェントノードの名前が蓄積されます。
2. (タスク 2.5 より) 正しい langgraph パッケージから InMemorySaver をインポートし、`InMemorySaver` を使ったチェックポインター付きでワークフローをコンパイルする。チェックポインターは呼び出しをまたいで状態を永続化するため、ワークフローを複数回呼び出してもアシスタントは以前の状態を記憶します。`create_workflow` を変更して `workflow.compile(checkpointer=InMemorySaver())` を呼び出すようにしてください。`InMemorySaver` のインポートが必要です。
3. `assistant.py` の `process_message` メソッドで、`config` オブジェクト内の `configurable` の値を正しく設定する必要があります。具体的には以下を設定してください:
   - `thread_id` を current_sessions.session_id に
   - `llm` を設定済みの LLM インスタンスに
   - `tools`

これらを追加することで、エージェントの処理の流れを追跡し、永続的な状態を試せるようになります。例については、状態管理とメモリのデモ演習を参照してください。

### 3. プロンプトの実装 (prompts.py)

#### タスク 3.1: チャットプロンプトテンプレート

`prompts.py` の `get_chat_prompt_template` 関数を完成させてください。

1. "qa"、"summarization"、"calculation" のいずれかとなる `intent_type` パラメータのすべてに対応するよう関数の実装を仕上げる
2. prompts.py を確認してファイル内のすべてのプロンプトを把握したうえで、`get_chat_prompt_template` 関数が `intent_type` パラメータに基づいて system_prompt を正しい値に設定するようにする
   > ファイル内ですでに定義されている既存のプロンプト (QA_SYSTEM_PROMPT、SUMMARIZATION_SYSTEM_PROMPT、CALCULATION_SYSTEM_PROMPT) を必ず使用してください

**目的**: タスクの種類ごとに、コンテキストに応じたプロンプトを提供します。

#### タスク 3.2: CalculationSystemPrompt の実装

`prompts.py` の `CALCULATION_SYSTEM_PROMPT` 定数を実装してください。

1. 計算エージェント用のシステムプロンプトを書き、LLM に以下を指示する:

- 取得すべきドキュメントを判断し、document reader ツールを使って取得する
- ユーザーの入力に基づいて計算すべき数式を判断する
- calculator ツールを使って計算を実行する

2. どんなに単純な計算であっても、LLM が「すべての」計算で calculator ツールを使うようにする

### 4. ツールの実装 (tools.py)

#### タスク 4.1: 計算ツール

以下を満たす `create_calculator_tool` 関数を実装してください。

1. `@tool` デコレーターを使って LangChain ツールを作成する
2. 数式を入力として受け取る
3. 安全性のために式を検証する(基本的な算術演算のみ許可する)
4. Python の `eval()` 関数で式を評価する
5. ToolLogger でツールの使用をログに記録する
6. 整形された結果文字列を返す
7. エラーを適切に処理する

## 成功のための重要な概念

### 1. LangChain のツールパターン

ツールは `@tool` でデコレートされた、LLM から呼び出せる関数です。ツールは以下を満たす必要があります。

- 目的とパラメータを説明する明確な docstring を持つ
- エラーを適切に処理する
- 結果を文字列で返す
- デバッグのために使用状況をログに記録する

### 2. LangGraph の状態管理

状態はノード間を流れ、各ステップで更新されます。主な原則:

- ノード関数からは常に更新された状態を返す
- ノード間の情報の受け渡しには状態を使う
- 状態は会話のコンテキストと中間結果を保持する

### 3. 構造化出力

文字列をパースする代わりに、`llm.with_structured_output(YourSchema)` を使って LLM から信頼性の高い型付きの応答を得ます。

### 4. 会話メモリ

システムは InMemorySaver チェックポインターによって会話を維持します。

- メタデータ付きで会話メッセージを保存する
- アクティブなドキュメントを追跡する
- 会話を要約する
- 後続のリクエストにコンテキストを提供する

## 実装のテスト

1. **ユニットテスト**: サンプル入力で個々の関数をテストする
2. **統合テスト**: さまざまなユーザー入力でワークフロー全体をテストする
3. **エッジケース**: エラー処理とエッジケースをテストする

## よくある落とし穴

1. **エラー処理の欠如**: 外部呼び出しは必ず try-catch ブロックで囲む
2. **誤った状態更新**: 状態を正しく更新して返していることを確認する
3. **プロンプトエンジニアリング**: プロンプトが明確かつ具体的であることを確認する
4. **ツールのセキュリティ**: セキュリティ上の問題を防ぐため、すべての入力を検証する

## 期待される動作

実装後、アシスタントは以下ができるようになっているはずです。

- ユーザーの意図を正しく分類する
- 関連するドキュメントを検索・取得する
- 適切な出典を示して質問に回答する
- 包括的な要約を生成する
- ドキュメントのデータに対して計算を行う
- ターンをまたいで会話のコンテキストを維持する

実装頑張ってください！十分にテストを行い、パターンやベストプラクティスについては既存の動作するコードを参考にしてください。
