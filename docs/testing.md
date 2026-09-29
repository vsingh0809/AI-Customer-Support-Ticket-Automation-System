# Testing & Verification

## AI Customer Support & Ticket Automation System

This document records the testing strategy, required functional scenarios, failure handling, and final acceptance checks for the application.

> **Note:** Final automated test counts must be copied from the latest successful `uv run pytest` execution. Do not replace the values with historical counts from an earlier run.

---

## 1. Automated Test Execution

Run the complete automated test suite from the project root:

```bash
uv run pytest
```

### Final Result

Record the output from the final clean run here:

```text
pytest result:
[PASTE FINAL COMMAND OUTPUT HERE]
```

### Acceptance Criteria

- All automated tests pass.
- No test is skipped without a documented reason.
- No existing failing test remains unresolved.

---

## 2. Static Analysis / Code Quality

Run:

```bash
uv run ruff check .
```

### Final Result

Record the output from the final clean run here:

```text
ruff result:
[PASTE FINAL COMMAND OUTPUT HERE]
```

### Acceptance Criteria

```text
Ruff → All checks passed!
```

No unresolved Ruff errors should remain.

---

## 3. Functional Test Scenarios

The assignment requires coverage for FAQ, knowledge-base questions, refund questions, order status, payment status, support-ticket creation, human escalation, unknown questions, invalid order IDs, tool failure, retrieval failure, conversation follow-up, and multiple requests. fileciteturn162file3

| # | Scenario | Input / Example | Expected Result | Status |
|---|---|---|---|---|
| 1 | General FAQ | Common customer-support question | Appropriate answer is returned | PASS / VERIFY |
| 2 | Knowledge-base question | Ask about a NovaMart policy | RAG retrieves relevant KB content and returns a grounded answer | PASS / VERIFY |
| 3 | Refund question | “What is your refund policy?” | Refund-policy information is retrieved from KB | PASS / VERIFY |
| 4 | Order status | “Where is my order 45821?” | `check_order_status` is executed and authoritative order information is returned | PASS / VERIFY |
| 5 | Payment status | Ask payment status for order `45821` | `check_payment_status` is executed and authoritative payment information is returned | PASS / VERIFY |
| 6 | Support ticket | Request support for an issue | Missing information is requested when necessary; otherwise ticket is created | PASS / VERIFY |
| 7 | Human escalation | Explicitly request human support | `escalate_to_human` creates an urgent escalation ticket | PASS / VERIFY |
| 8 | Unknown question | Unsupported / ambiguous request | System asks for clarification rather than inventing an answer | PASS / VERIFY |
| 9 | Invalid order ID | “Where is order 9999?” | Safe not-found/error response; no fabricated order information | PASS / VERIFY |
| 10 | Tool failure | Simulate business-tool failure | Deterministic customer-safe fallback is returned | PASS / VERIFY |
| 11 | Retrieval failure | Simulate RAG/retrieval failure | Retrieval failure is handled without fabricated KB content | PASS / VERIFY |
| 12 | Conversation follow-up | “My order number is 45821” → “When will it arrive?” | Previous conversation context is retained and used | PASS / VERIFY |
| 13 | Multiple requests | Order status + payment status in one message | Planner produces multiple actions and both requests are addressed | PASS / VERIFY |

---

## 4. RAG Verification

### Objective

Confirm that company-specific questions use the knowledge base rather than relying only on general LLM knowledge.

### Verification Flow

```text
Customer Query
      ↓
Intent Classification
      ↓
Action Planning
      ↓
RAG Workflow
      ↓
Embedding / Retrieval
      ↓
Qdrant
      ↓
Relevant Knowledge Chunks
      ↓
LLM
      ↓
Grounded Response
```

### Checks

- Knowledge-base documents are present.
- Documents are chunked and embedded.
- Embeddings are stored in the `support_kb` Qdrant collection.
- Retrieval returns relevant chunks.
- Generated responses are grounded in retrieved context.
- Source information is returned where supported by the RAG workflow.

---

## 5. Tool Calling Verification

Implemented business tools:

```text
check_order_status
check_payment_status
create_support_ticket
escalate_to_human
```

### Required Checks

Each tool should be verified for:

```text
Input validation
      ↓
Customer ownership / authorization checks
      ↓
Application service
      ↓
Repository
      ↓
PostgreSQL
      ↓
Authoritative result
      ↓
Customer response
```

The tools perform actual application logic rather than returning only fixed demonstration text.

---

## 6. Multi-Tool Verification

### Example

```text
Customer:
Check my order 45821 and tell me whether the payment was successful.
```

Expected workflow:

```text
Intent Classification
        ↓
Action Planner
        ↓
┌──────────────────────────────┐
│ check_order_status           │
│ check_payment_status         │
└──────────────────────────────┘
        ↓
Execute both actions
        ↓
Collect authoritative results
        ↓
Generate one coherent response
```

### Failure Variant

Simulate one successful tool and one failed tool.

Expected behavior:

- Preserve successful authoritative information.
- Clearly state which action could not be completed.
- Do not fabricate the failed result.
- Return a non-empty customer response.

---

## 7. Mixed RAG + Tool Verification

The system supports workflows where a customer request requires both knowledge retrieval and an application action.

Conceptual flow:

```text
Customer Request
      ↓
Intent Classification
      ↓
Action Planner
      ↓
RAG Action + Tool Action
      ↓
Execute planned actions
      ↓
Combine results
      ↓
Final customer response
```

### Acceptance Criteria

- The planner determines the required actions.
- RAG handles knowledge retrieval.
- Business tools handle authoritative transactional data.
- The final response does not invent unavailable information.

---

## 8. Conversation Memory Verification

### First Turn

```text
Customer:
My order number is 45821.
```

### Follow-Up

```text
Customer:
When will it arrive?
```

### Expected Behavior

The second request is evaluated in the context of the existing conversation.

The conversation memory flow is:

```text
Chat Request
      ↓
Load Conversation
      ↓
Load Previous Messages
      ↓
Agent receives Conversation History
      ↓
Process Current Request
      ↓
Persist User + Assistant Turns
```

### Persistence Check

Verify that:

- The conversation has a database record.
- Messages remain available after a frontend rerun.
- Selecting a previous conversation reloads its messages.
- Starting a new chat does not reuse the previous conversation ID.

---

## 9. Support Ticket Verification

### Standard Ticket

Expected workflow:

```text
Customer Support Request
        ↓
Intent Classification
        ↓
Action Planning
        ↓
Required Arguments
        ↓
Clarification when required
        ↓
create_support_ticket
        ↓
TicketService
        ↓
PostgreSQL
        ↓
Ticket Result
        ↓
Customer Response
```

### Acceptance Criteria

- Required ticket information is collected.
- Invalid or incomplete requests result in clarification.
- Successfully created tickets receive an authoritative ticket ID.
- Ticket creation failures return a safe fallback response.

---

## 10. Human Escalation Verification

### Example

```text
I want to speak with a human support agent.
```

Expected workflow:

```text
Human Escalation Intent
        ↓
Action Planner
        ↓
escalate_to_human
        ↓
Urgent Ticket Creation
        ↓
PostgreSQL
        ↓
Escalation Confirmation
```

### Acceptance Criteria

- Human escalation is identified.
- The escalation tool is invoked.
- An urgent support ticket is created.
- The customer receives a confirmation response.
- Failure is handled through a deterministic fallback.

---

## 11. Error Handling Verification

The application is required to handle invalid input, invalid order IDs, missing information, tool failures, LLM/API failures, retrieval failures, empty knowledge-base results, invalid tool parameters, and ticket creation failures. fileciteturn162file3

| Failure Type | Expected Handling |
|---|---|
| Invalid customer input | Validation error / safe response |
| Missing required argument | Clarification question |
| Invalid order ID | Safe order lookup failure |
| Invalid tool parameters | Validation failure |
| Tool failure | Deterministic customer-safe fallback |
| LLM failure | Appropriate fallback / application error handling |
| Retrieval failure | Safe retrieval failure handling |
| Empty KB result | No fabricated company-specific answer |
| Ticket creation failure | Safe ticket failure response |

The key safety requirement is that application failures must not cause the assistant to invent transactional or company-specific information.

---

## 12. Frontend Verification

The Streamlit frontend should be checked manually.

### Chat

- Chat input is displayed.
- User messages are rendered.
- Assistant responses are rendered.
- Loading state is displayed while the request is running.
- API errors are visible to the user.

### Persistent Conversation Sidebar

Verify:

```text
＋ New chat
        ↓
Recent conversations
        ↓
Select an existing conversation
        ↓
Load persisted messages
```

### New Conversation

Expected behavior:

```text
Current conversation
        ↓
New chat
        ↓
conversation_id = None
messages = []
        ↓
First new message creates a new conversation
```

Customer identity remains unchanged.

---

## 13. API Verification

Primary customer-facing interface:

```http
POST /chat
```

Expected request structure:

```json
{
  "customer_id": "3f2b8f18-7f3d-4d6e-9a2c-5c3e4a7d8b1f",
  "message": "Where is my order 45821?",
  "conversation_id": null
}
```

Expected response structure:

```json
{
  "conversation_id": "conversation-uuid",
  "response": "..."
}
```

Conversation-history APIs:

```http
GET /conversations?customer_id=<UUID>
GET /conversations/{conversation_id}?customer_id=<UUID>
```

---

## 14. Security / Configuration Checks

Before final submission, verify:

```text
.env is not committed
API keys are not committed
Provider credentials are not committed
Secrets are not hard-coded
Customer-specific data is ownership-checked
Production business operations are controlled by application code
```

The assignment explicitly requires secure credential management and prohibits committing credentials or API keys to GitHub. fileciteturn162file3

---

## 15. Final Acceptance Checklist

### Core Functionality

- [ ] Customer can use the Streamlit chat interface.
- [ ] FastAPI `/chat` endpoint works.
- [ ] Knowledge-base questions use RAG.
- [ ] Order status uses a real business tool.
- [ ] Payment status uses a real business tool.
- [ ] Support ticket creation works.
- [ ] Human escalation works.
- [ ] Multiple requests can be handled.
- [ ] Mixed RAG + tool workflows work.

### Memory

- [ ] Multi-turn context works.
- [ ] Conversations persist in PostgreSQL.
- [ ] Existing conversations can be reopened.
- [ ] New conversation starts cleanly.

### Reliability

- [ ] Missing arguments trigger clarification.
- [ ] Invalid order IDs are handled safely.
- [ ] Tool failures have safe fallbacks.
- [ ] Retrieval failures are handled.
- [ ] LLM failures are handled.
- [ ] Ticket failures are handled.
- [ ] No fabricated transactional information is returned.

### Quality

- [ ] `uv run pytest` passes.
- [ ] `uv run ruff check .` passes.
- [ ] No secrets are committed.
- [ ] README is complete.
- [ ] Architecture diagram is present.
- [ ] Test results are documented.

---

## 16. Final Demonstration Coverage

The assignment requires the final demonstration to show the complete system, including RAG, action/tool calling, multiple tools, conversation memory, ticket creation, human escalation, an error scenario, and technical architecture explanation. fileciteturn162file0

Recommended demonstration order:

```text
1. Open Streamlit application
        ↓
2. Ask a knowledge-base question
        ↓
3. Explain RAG retrieval
        ↓
4. Ask order-status question
        ↓
5. Explain tool execution
        ↓
6. Ask order + payment together
        ↓
7. Explain multiple-tool planning
        ↓
8. Demonstrate follow-up conversation
        ↓
9. Create support ticket
        ↓
10. Demonstrate human escalation
        ↓
11. Demonstrate invalid order / failure case
        ↓
12. Show persisted conversations
        ↓
13. Explain architecture
```

This sequence covers the major capabilities requested in the assignment's final demonstration criteria. fileciteturn162file3

---

## 17. Final Evidence to Collect

For the final submission, collect:

```text
Source code
GitHub repository
Working application/API
Knowledge-base documents
README
Architecture diagram
Test cases
Test results
Sample customer queries
Screenshots
Demo video
Final presentation
```

These items are explicitly listed in the assignment's final-submission requirements. fileciteturn162file0

---

## 18. Test Result Recording

After the final regression run, replace the placeholders in this document with the actual command outputs:

```bash
uv run pytest
uv run ruff check .
```

Do not manually estimate the final test count.

---

## 19. Completion Standard

The project is considered technically complete when the implemented system demonstrates:

```text
RAG
  ↓
Agent
  ↓
Tools
  ↓
Conversation Memory
  ↓
API
  ↓
Testing
  ↓
Documentation
  ↓
Demonstration
```

This matches the completion standard specified by the project assignment. fileciteturn162file0
