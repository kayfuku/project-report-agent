# DocDacity Document Assistant

A multi-agent document processing assistant built with LangChain and LangGraph. It handles three kinds of requests about financial and healthcare documents such as invoices, contracts, and insurance claims: answering questions (Q&A), summarizing content, and performing calculations on the numbers in the documents.

Each user input is first classified by the LLM into an intent, and then routed to a dedicated agent based on that intent. Each agent runs as a ReAct agent that uses tools (document search, document reader, document statistics, and calculator) and returns its answer as structured output defined by a Pydantic schema. Finally, a conversation summary and the documents currently being discussed are stored in memory, and the `InMemorySaver` checkpointer keeps the state across multiple turns within the same session.

![Agent architecture](./starter/docs/langgraph_agent_architecture.png)

```
classify_intent --> [qa | summarization | calculation] --> update_memory --> END
```

## Getting Started

Follow the steps below to run the assistant on your local machine. All commands are run from the `starter/` directory.

### Dependencies

- Python 3.9+ (Python 3.11 was used for development)
- An OpenAI API key (or an endpoint for an OpenAI-compatible API)
- The packages listed in [starter/requirements.txt](starter/requirements.txt)

```
langchain>=0.2.0
langgraph>=0.6.7
langchain-openai>=0.1.0
langchain-core>=0.2.0
pydantic>=2.0.0
python-dotenv>=1.0.0
openai>=1.0.0
print-color>=0.4.6
pytest>=8.0.0
```

### Installation

1. Move to the `starter` directory.

```bash
cd project/starter
```

2. Create and activate a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies.

```bash
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and set the values.

```bash
cp .env.example .env
```

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | Your OpenAI API key. The assistant exits with an error at startup if it is not set |
| `MODEL_NAME` | Yes | The model to use (e.g. `gpt-4o`) |
| `BASE_URL` | No | The endpoint for an OpenAI-compatible API. It is not included in `.env.example`, so add it if you need it |

5. Start the assistant.

```bash
python main.py
```

On startup you are asked for a user ID (press Enter for `demo_user`). After you enter a message, the assistant shows its answer together with the classified intent, the documents it referenced, the tools it used, and a summary of the conversation.

```
/help  - Show the help message
/docs  - List available documents
/quit  - Exit the assistant

Example queries:
  - What's the total amount in invoice INV-001?
  - Summarize all contracts
  - Calculate the sum of all invoice totals
  - Find documents with amounts over $50,000
```

Conversation sessions are saved to `sessions/` and tool usage logs to `logs/`, both as JSON.

## Implementation Details

### Implementation Decisions

- **Node names match the intent values**: The graph nodes are named `qa` / `summarization` / `calculation` so that they match the `next_step` value set by `classify_intent`. This keeps the conditional edge mapping simple. `actions_taken` records agent names such as `qa_agent`.
- **Unknown intents go to Q&A**: If the intent is `unknown` or any other unexpected value, the request falls back to the default Q&A agent.
- **One pattern for all agent nodes**: All three agents follow the same flow (get the intent-specific prompt → run the ReAct agent → update the state) and differ only in the system prompt and the output schema. The shared part lives in `invoke_react_agent`.
- **Calculations always use the calculator tool**: `CALCULATION_SYSTEM_PROMPT` tells the LLM to use the calculator tool for every calculation, no matter how simple, and never to calculate on its own.
- **Calculator safety**: The calculator strips `$` and `,`, uses a regular expression to allow only digits, whitespace, and `+ - * / % ( ) .`, and runs `eval()` with empty `__builtins__`. Invalid expressions and division by zero return an error message instead of raising, and are logged.
- **Tests without an LLM**: The tests replace the LLM and the ReAct agent with stubs, so the whole workflow can be checked deterministically without an API key.

### State and Memory Behavior

- **State (`AgentState`)**: `messages` uses the `add_messages` reducer and `actions_taken` uses the `operator.add` reducer, so both are appended to rather than overwritten. The other fields (`intent`, `next_step`, `current_response`, and so on) are overwritten by the values each node returns.
- **Checkpointer**: The workflow is compiled with `InMemorySaver`, and the session ID is passed as `thread_id`. Within the same session the state from earlier turns carries over, so `messages` (the conversation history) and `actions_taken` build up turn by turn.
- **Conversation memory (`update_memory`)**: At the end of each turn, the LLM summarizes all messages so far and updates `conversation_summary` (the conversation summary) and `active_documents` (the document IDs relevant to the latest exchange).
- **Saving sessions**: `DocumentAssistant` saves the final state of each turn to `sessions/<session_id>.json` and adds the referenced document IDs to `document_context` as a union.
- **Limitation**: `InMemorySaver` keeps its state only in process memory, so the conversation context is lost when the app restarts. The session JSON remains as a record, but the checkpointer state is not restored from it.

### How Structured Output Is Enforced

- **Intent classification and memory updates**: `llm.with_structured_output(UserIntent)` and `llm.with_structured_output(UpdateMemoryResponse)` return the LLM output directly as Pydantic models. `UserIntent.intent_type` is `Literal["qa", "summarization", "calculation", "unknown"]`, so no other value is accepted.
- **Agent answers**: `AnswerResponse` / `SummarizationResponse` / `CalculationResponse` is passed to `create_agent(..., response_format=schema)`. LangChain presents the schema to the LLM as a "final answer tool", so once the agent is done with its tools it returns the answer in the shape of the schema. The result is stored in `current_response["structured_response"]`. This is also why `tools_used` includes the schema name (e.g. `CalculationResponse`).
- **Validation in the schemas**: `confidence` is limited to `0.0–1.0` (`AnswerResponse`), and `sources` and `timestamp` have default values, so a response is still valid when the LLM leaves some fields out.

## Example Conversation

This is a real record of one 4-turn session by `user1` ([sessions/9273134b-bfa3-4a7f-9bfc-7940776ce031.json](starter/sessions/9273134b-bfa3-4a7f-9bfc-7940776ce031.json)). It shows all three features (summarization, calculation, and Q&A), tool use, structured output, and memory across turns. The responses are excerpts.

**Turn 1: Summarization**

```
Enter Message: Summarize all contracts

🤖 Assistant: Returning structured response: original_length=844
  summary="There is 1 contract on file. CON-001 — Service Agreement
  - Parties: DocDacity Solutions Inc. (Provider) and Healthcare Partners LLC (Client) ..."
  key_points=['One contract found: CON-001', 'Total value: $180,000', 'Monthly fee: $15,000',
              'Term: 12 months', "Termination requires 60 days' written notice", ...]
  document_ids=['CON-001']

INTENT: summarization
SOURCES: CON-001
TOOLS USED: document_search, document_reader, SummarizationResponse
```

**Turn 2: Calculation**

```
Enter Message: Calculate the sum of all invoice totals

🤖 Assistant: Returning structured response: expression='20000 + 69300 + 214500' result=303800.0
  explanation='Summed the invoice totals from documents INV-001, INV-002, and INV-003:
  $20,000 + $69,300 + $214,500 = $303,800.' units='USD'

INTENT: calculation
SOURCES: CON-001, INV-001, INV-002, INV-003
TOOLS USED: ..., document_search, document_reader, document_reader, document_reader, calculator, CalculationResponse
```

**Turn 3: Q&A (search by amount)**

```
Enter Message: Find documents with amounts over $50,000

🤖 Assistant: Returning structured response: question='Find documents with amounts over $50,000'
  answer='Found 3 documents with amounts over $50,000:
  - INV-003 — Invoice #12347 — $214,500.00
  - CON-001 — Service Agreement — $180,000.00
  - INV-002 — Invoice #12346 — $69,300.00'
  sources=['INV-003', 'CON-001', 'INV-002'] confidence=0.99

INTENT: qa
TOOLS USED: ..., document_search, AnswerResponse
```

**Turn 4: Calculation (checking memory)**

```
Enter Message: What's the total amount in invoice INV-001?

🤖 Assistant: Returning structured response: expression='20000 + 2000' result=22000.0
  explanation='Using invoice INV-001, the subtotal is $20,000 and the tax is $2,000,
  so the total amount is $20,000 + $2,000 = $22,000.' units='USD'

INTENT: calculation
CONVERSATION SUMMARY: Conversation covered document summarization and financial analysis ...
  - Sum of all invoice totals: INV-001 $20,000 + INV-002 $69,300 + INV-003 $214,500 = $303,800.
  - Documents with amounts over $50,000: INV-003 ($214,500), CON-001 ($180,000), INV-002 ($69,300).
  - Total amount in invoice INV-001: subtotal $20,000 plus tax $2,000 = $22,000.
```

What this session shows:

- The conversation summary in turn 4 includes what happened in turns 1–3, so memory is kept across turns.
- After four turns, `actions_taken` is `['classify_intent', 'summarization_agent', 'classify_intent', 'calculation_agent', 'classify_intent', 'qa_agent', 'classify_intent', 'calculation_agent']`: `operator.add` has accumulated the nodes from every turn.
- Every calculation was done with the `calculator` tool.
- **Known limitation**: INV-001 has no "Total Due" line, so in turn 2 the subtotal of $20,000 was used as its amount in the sum. Turn 4 correctly calculated $22,000 including tax. Which value counts as the "total" depends on how the LLM reads the document.

## Testing

The tests are in [starter/tests/test_assistant.py](starter/tests/test_assistant.py). The LLM and the ReAct agent are replaced with stubs, so the tests run offline and deterministically without an API key. Run the following command from the `starter/` directory.

```bash
venv/bin/python -m pytest -v
```

All 18 tests currently pass.

### Break Down Tests

The tests are split into three groups, following "Testing Your Implementation" in the starter README.

**1. Unit tests**: check individual functions with sample inputs.

| Test | What it checks |
| --- | --- |
| `test_answer_response_defaults` | `AnswerResponse` can be built from just a question and an answer, and the other fields get default values |
| `test_user_intent_fields` | `UserIntent` holds the fields used for routing |
| `test_calculator` | The calculator respects operator precedence, accepts amounts written with `$` and `,`, and keeps decimals |
| `test_calculator_logs_usage` | Tool calls are recorded by `ToolLogger` |
| `test_document_reader_found` | `document_reader` returns the full content of an existing document |
| `test_document_search_over_amount` | An amount search ("over $50,000") returns only the documents above the threshold |

**2. Integration tests**: run the complete workflow.

| Test | What it checks |
| --- | --- |
| `test_routes_by_intent_and_updates_memory` | A calculation intent runs `classify_intent → calculation → update_memory` and updates the state with the correct schema |
| `test_qa_workflow` | A qa intent is routed to the Q&A agent and answered with `AnswerResponse` |
| `test_summarization_workflow` | A summarization intent is routed to the summarization agent and answered with `SummarizationResponse` |
| `test_actions_accumulate_across_turns` | With the checkpointer and the `operator.add` reducer, `actions_taken` accumulates across turns in the same thread |
| `test_unknown_intent_falls_back_to_qa` | An unexpected intent (unknown) falls back to the default Q&A agent |

**3. Edge cases**: check error handling.

| Test | What it checks |
| --- | --- |
| `test_calculator_invalid_expression` | Code injection, division by zero, and empty expressions return an error message instead of raising or executing code |
| `test_document_reader_missing_id` | An unknown document ID returns a "not found" message |
| `test_document_search_no_results` | A search with no matches returns a "no documents" message |

```
$ venv/bin/python -m pytest -q
..................                                                       [100%]
18 passed
```

## Project Instructions

The full details of the tasks for students are in [starter/README.md](starter/README.md) (Japanese version: [starter/README_ja.md](starter/README_ja.md)). The tasks are:

| Task | File | Description |
| --- | --- | --- |
| 1.1 | `src/schemas.py` | `AnswerResponse` schema for structured Q&A output |
| 1.2 | `src/schemas.py` | `UserIntent` schema for intent classification |
| 2.1 | `src/agent.py` | Understanding the structure of `AgentState` |
| 2.2 | `src/agent.py` | `classify_intent` node (classifies the intent with structured output and sets `next_step`) |
| 2.3 | `src/agent.py` | `summarization_agent` and `calculation_agent` nodes |
| 2.4 | `src/agent.py` | `update_memory` node (updates the conversation summary and active documents) |
| 2.5 | `src/agent.py` | Building the graph in `create_workflow` (nodes, conditional edges, edges) |
| 2.6 | `src/agent.py`, `src/assistant.py` | `operator.add` reducer on `actions_taken`, compiling with `InMemorySaver`, and setting `config` in `process_message` |
| 3.1 | `src/prompts.py` | `get_chat_prompt_template`, which selects the system prompt based on the intent |
| 3.2 | `src/prompts.py` | `CALCULATION_SYSTEM_PROMPT`, which instructs the LLM to always use the calculator tool |
| 4.1 | `src/tools.py` | Calculator tool that validates the input, evaluates the expression, and logs its usage |

### Project Structure

```
starter/
├── src/
│   ├── schemas.py        # Pydantic models (structured output, session state)
│   ├── retrieval.py      # Document retrieval (simulated, no vector database)
│   ├── tools.py          # Agent tools (search, reader, statistics, calculator)
│   ├── prompts.py        # Prompt templates
│   ├── agent.py          # LangGraph workflow
│   └── assistant.py      # Session management and workflow invocation
├── tests/                # pytest tests
├── docs/                 # Architecture diagram
├── sessions/             # Saved conversation sessions
├── logs/                 # Tool usage logs
├── main.py               # Entry point (interactive CLI)
└── requirements.txt      # Dependencies
```

### Sample Documents

`SimulatedRetriever` loads the following five sample documents into memory.

| ID | Type | Content |
| --- | --- | --- |
| `INV-001` | Invoice | Billed to Acme Corporation, subtotal $20,000 |
| `INV-002` | Invoice | Billed to TechStart Inc., total $69,300 |
| `INV-003` | Invoice | Billed to Global Corp, total $214,500 |
| `CON-001` | Contract | 12-month service agreement with Healthcare Partners LLC, total value $180,000 |
| `CLM-001` | Insurance claim | Medical expense reimbursement claim, claim amount $2,450 |

## Built With

* [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent workflow (state graph, conditional routing, checkpointer)
* [LangChain](https://python.langchain.com/) - ReAct agents, tools (`@tool`), prompt templates
* [LangChain OpenAI](https://python.langchain.com/docs/integrations/chat/openai/) - LLM calls through `ChatOpenAI`
* [OpenAI API](https://platform.openai.com/docs/) - LLM for intent classification, answer generation, and summarization
* [Pydantic](https://docs.pydantic.dev/) - Schemas for structured output and session state
* [python-dotenv](https://github.com/theskumar/python-dotenv) - Loading environment variables from `.env`
* [print-color](https://pypi.org/project/print-color/) - Colored CLI output
* [pytest](https://docs.pytest.org/) - Test framework

## License
[License](../LICENSE.md)
