"""
Simple tests following README.md "Testing Your Implementation":
1. Unit Testing        - individual functions with sample inputs
2. Integration Testing - the complete workflow (LLM is stubbed, no API key needed)
3. Edge Cases          - error handling and edge cases

Run from the starter directory:
    venv/bin/python -m pytest -v
"""

import pytest
from langchain_core.messages import AIMessage

from src import agent
from src.agent import create_workflow
from src.retrieval import SimulatedRetriever
from src.schemas import (
    AnswerResponse, CalculationResponse, SummarizationResponse,
    UpdateMemoryResponse, UserIntent
)
from src.tools import (
    ToolLogger, create_calculator_tool, create_document_reader_tool,
    create_document_search_tool
)


class FakeStructuredLLM:
    """Stub LLM: returns a canned response for each schema passed to with_structured_output.

    Mimics llm.with_structured_output(schema).invoke(prompt) (duck typing), so tests
    run without an API key and get deterministic results. The prompt is ignored.
    An unregistered schema raises KeyError.
    """

    def __init__(self, responses):
        self.responses = responses  # {schema class: response instance}

    def with_structured_output(self, schema):
        llm = self  # keep a reference; _Runner.invoke has its own `self`

        class _Runner:
            def invoke(self, _prompt):
                return llm.responses[schema]

        return _Runner()


@pytest.fixture
def logger(tmp_path):
    return ToolLogger(logs_dir=str(tmp_path))


@pytest.fixture
def retriever():
    return SimulatedRetriever()


@pytest.fixture
def calculator(logger):
    return create_calculator_tool(logger)


@pytest.fixture
def reader(retriever, logger):
    return create_document_reader_tool(retriever, logger)


@pytest.fixture
def search(retriever, logger):
    return create_document_search_tool(retriever, logger)


# ---------------------------------------------------------------------------
# 1. Unit Testing
# ---------------------------------------------------------------------------
def test_answer_response_defaults():
    """AnswerResponse can be built from question/answer alone; optional fields get defaults."""
    response = AnswerResponse(question="Q?", answer="A.")
    assert response.sources == []
    assert response.confidence == 0.0
    assert response.timestamp is not None


def test_user_intent_fields():
    """UserIntent holds the fields used by classify_intent for routing."""
    intent = UserIntent(intent_type="calculation", confidence=0.9,
                        reasoning="User asks for a sum")
    assert intent.intent_type == "calculation"
    assert intent.confidence == 0.9


@pytest.mark.parametrize("expression, expected", [
    ("2 + 3 * 4", "2 + 3 * 4 = 14"),
    ("$20,000 + $2,000", "$20,000 + $2,000 = 22000"),
    ("10 / 4", "10 / 4 = 2.5"),
])
def test_calculator(calculator, expression, expected):
    """Calculator respects operator precedence, accepts "$" and "," as in document amounts, and keeps decimals."""
    assert calculator.invoke({"expression": expression}) == expected


def test_calculator_logs_usage(calculator, logger):
    """Tool calls are recorded by ToolLogger (used for the session log)."""
    calculator.invoke({"expression": "1 + 1"})
    assert logger.get_logs()[-1]["tool_name"] == "calculator"


def test_document_reader_found(reader):
    """document_reader returns the full content of an existing document."""
    assert "Invoice #12345" in reader.invoke({"doc_id": "INV-001"})


def test_document_search_over_amount(search):
    """Amount search ("over $50,000") returns only documents above the threshold."""
    result = search.invoke({"query": "documents over $50,000",
                            "search_type": "amount",
                            "comparison": "over", "amount": 50000})
    for doc_id in ["INV-002", "INV-003", "CON-001"]:
        assert doc_id in result
    assert "INV-001" not in result


# ---------------------------------------------------------------------------
# 2. Integration Testing
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_llm():
    """Stub LLM for the structured-output nodes (classify_intent, update_memory)"""
    return FakeStructuredLLM({
        # classify_intent -> routes to calculation_agent
        UserIntent: UserIntent(intent_type="calculation", confidence=0.95),
        # update_memory -> sets conversation_summary / active_documents
        UpdateMemoryResponse: UpdateMemoryResponse(
            summary="User asked for the total of INV-001.",
            document_ids=["INV-001"]),
    })


@pytest.fixture
def react_calls(monkeypatch):
    """Replace the ReAct agent so no real LLM / tool calls are made.

    Used by the agent nodes (qa_agent, summarization_agent, calculation_agent).
    Acts as a stub (canned result) and a mock (records each response_schema).
    """
    calls = []  # response_schema of each call, checked by the tests
    # {schema: (result, tools_used)}; result has the same shape as the real agent.invoke() result
    fake_results = {
        AnswerResponse: ({
            "messages": [AIMessage(content="INV-002 is billed to TechStart Inc.")],
            "structured_response": AnswerResponse(
                question="Who is INV-002 billed to?",
                answer="TechStart Inc.",
                sources=["INV-002"], confidence=0.9),
        }, ["document_search"]),
        SummarizationResponse: ({
            "messages": [AIMessage(content="CON-001 is a 12-month service agreement with Healthcare Partners LLC.")],
            "structured_response": SummarizationResponse(
                original_length=661,  # len() of CON-001 content
                summary="12-month service agreement between DocDacity Solutions Inc. "
                        "and Healthcare Partners LLC for a document processing platform.",
                key_points=["Duration: 12 months", "Monthly Fee: $15,000",
                            "Total Contract Value: $180,000",
                            "Termination with 60 days written notice"],
                document_ids=["CON-001"]),
        }, ["document_reader"]),
        CalculationResponse: ({
            "messages": [AIMessage(content="The total is 22000.")],
            "structured_response": CalculationResponse(
                expression="20000 + 2000", result=22000,
                explanation="Subtotal plus tax"),
        }, ["calculator"]),
    }

    # Same signature and return type (result, tools_used) as the real function
    def fake_invoke_react_agent(response_schema, messages, llm, tools):
        calls.append(response_schema)
        return fake_results[response_schema]

    # Nodes look up invoke_react_agent in src.agent at call time, so they get the fake.
    # monkeypatch restores the original after each test.
    monkeypatch.setattr(agent, "invoke_react_agent", fake_invoke_react_agent)
    return calls


def run_workflow(workflow, llm, user_input, thread_id="test-thread"):
    config = {"configurable": {"thread_id": thread_id,
                               "llm": llm, "tools": []}}
    return workflow.invoke({"user_input": user_input,
                            "messages": [],
                            "actions_taken": []}, config)


def test_routes_by_intent_and_updates_memory(fake_llm, react_calls):
    """End-to-end: classify_intent -> calculation -> update_memory, with the right schema and state updates."""
    workflow = create_workflow(fake_llm, [])
    state = run_workflow(workflow, fake_llm, "What is the total of INV-001?")

    assert state["actions_taken"] == ["classify_intent", "calculation_agent"]
    assert state["tools_used"] == ["calculator"]
    assert state["current_response"]["structured_response"].result == 22000
    assert state["conversation_summary"] == "User asked for the total of INV-001."
    assert state["active_documents"] == ["INV-001"]
    # The calculation node must ask for the CalculationResponse schema
    assert react_calls == [CalculationResponse]


def test_qa_workflow(fake_llm, react_calls):
    """End-to-end: classify_intent -> qa -> update_memory, answered with AnswerResponse."""
    fake_llm.responses[UserIntent] = UserIntent(intent_type="qa", confidence=0.9)
    workflow = create_workflow(fake_llm, [])
    state = run_workflow(workflow, fake_llm, "Who is INV-002 billed to?")

    assert state["actions_taken"] == ["classify_intent", "qa_agent"]
    assert state["tools_used"] == ["document_search"]
    response = state["current_response"]["structured_response"]
    assert response.answer == "TechStart Inc."
    assert response.sources == ["INV-002"]
    assert react_calls == [AnswerResponse]


def test_summarization_workflow(fake_llm, react_calls):
    """End-to-end: classify_intent -> summarization -> update_memory, answered with SummarizationResponse."""
    fake_llm.responses[UserIntent] = UserIntent(
        intent_type="summarization", confidence=0.9)
    workflow = create_workflow(fake_llm, [])
    state = run_workflow(workflow, fake_llm, "Summarize CON-001")

    assert state["actions_taken"] == ["classify_intent", "summarization_agent"]
    assert state["tools_used"] == ["document_reader"]
    response = state["current_response"]["structured_response"]
    assert "Total Contract Value: $180,000" in response.key_points
    assert response.document_ids == ["CON-001"]
    assert react_calls == [SummarizationResponse]


def test_actions_accumulate_across_turns(fake_llm, react_calls):
    """The checkpointer keeps state per thread, so actions_taken grows across turns."""
    workflow = create_workflow(fake_llm, [])
    run_workflow(workflow, fake_llm, "First question")
    state = run_workflow(workflow, fake_llm, "Second question")
    # operator.add reducer keeps the history within the same thread
    assert state["actions_taken"].count("classify_intent") == 2


def test_unknown_intent_falls_back_to_qa(fake_llm, react_calls):
    """An unexpected intent is routed to the default qa agent with AnswerResponse."""
    fake_llm.responses[UserIntent] = UserIntent(intent_type="unknown")
    workflow = create_workflow(fake_llm, [])
    state = run_workflow(workflow, fake_llm, "Hello?")
    assert "qa_agent" in state["actions_taken"]
    assert react_calls == [AnswerResponse]


# ---------------------------------------------------------------------------
# 3. Edge Cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("expression", [
    "__import__('os').getcwd()",  # code injection
    "1 / 0",                      # division by zero
    "   ",                        # empty expression
])
def test_calculator_invalid_expression(calculator, expression):
    """Invalid or unsafe input returns an error message instead of raising or executing code."""
    assert calculator.invoke({"expression": expression}).startswith("Error")


def test_document_reader_missing_id(reader):
    """An unknown document ID returns a "not found" message, not an exception."""
    assert reader.invoke({"doc_id": "XXX-999"}) == \
        "Document with ID XXX-999 not found."


def test_document_search_no_results(search):
    """A search with no matches returns a clear "no documents" message."""
    assert search.invoke({"query": "zzzz-nonexistent-term"}) == \
        "No documents found matching your search criteria."
