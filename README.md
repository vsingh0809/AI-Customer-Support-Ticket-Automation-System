AI Customer Support & Ticket Automation System

An AI-powered customer-support application that combines:

Retrieval-Augmented Generation (RAG)

Agent-based workflow orchestration

Business tools

Conversation memory

Support ticket creation

Human escalation

FastAPI backend

Streamlit customer-facing interface

PostgreSQL persistence

Qdrant vector search

The system is designed to handle both knowledge-based questions and action-oriented customer requests rather than behaving as a simple LLM chatbot.

1. Project Overview

The AI Customer Support & Ticket Automation System automatically handles common customer-support workflows.

Customers can:

Ask general support questions

Ask company-policy questions

Check order status

Check payment status

Request support

Create support tickets

Continue previous conversations

Escalate issues to human support

The agent determines the appropriate workflow for each request and can combine multiple actions when required.

2. Problem Statement

Customer-support teams receive repetitive requests related to:

Orders

Payments

Refunds

Cancellations

Shipping

Product information

Account issues

Company policies

A traditional chatbot can answer questions but cannot reliably perform application actions.

This project addresses that limitation by separating:

Knowledge retrieval

Agent reasoning and workflow planning

Deterministic business operations

Persistent conversation memory

Ticket and escalation workflows

3. Objectives

The project aims to build a functional AI customer-support system that can:

Understand customer intent

Retrieve grounded information from a knowledge base

Execute real application tools

Ask for missing information

Maintain conversation context

Support multiple requests in one message

Create support tickets

Escalate unresolved cases

Handle application and AI failures safely

4. Features

Customer Support

Natural-language chat interface

Multi-turn conversations

Persistent conversation history

New conversation support

Knowledge Base

FAQ

Product information

Refund policy

Cancellation policy

Shipping policy

Payment policy

Account policy

Support guidelines

RAG

Markdown document loading

Document normalization

Chunking

Embedding generation

Qdrant vector storage

Similarity retrieval

Grounded response generation

Source references

Agent Workflows

The agent can decide to:

Search the knowledge base

Call a business tool

Ask for missing information

Create a support ticket

Escalate to a human

Combine multiple actions

Business Tools

Implemented tools:

check_order_status

check_payment_status

create_support_ticket

escalate_to_human

Reliability

Input validation

Tool validation

Customer ownership checks

Meaningful fallback responses

Retrieval failure handling

LLM failure handling

Tool failure handling

Ticket creation failure handling

5. Technology Stack

Backend

Python 3.11+

FastAPI

SQLAlchemy

Alembic

PostgreSQL 16

Pydantic

Pydantic Settings

AI / Agent

LangGraph

DeepSeek

Agent intent classification

Deterministic action planning

Tool calling

Multi-tool execution

RAG

FastEmbed

BAAI/bge-small-en-v1.5

Qdrant

Cosine similarity search

Frontend

Streamlit

HTTP / Testing / Tooling

HTTPX2

Pytest

Ruff

uv

Docker Compose

6. System Architecture

flowchart TD
    Customer[Customer]

    Frontend[Streamlit Frontend]
    API[FastAPI Backend]
    ChatService[Chat Application Service]
    Agent[LangGraph AI Agent]

    Classifier[Intent Classifier]
    Planner[Action Planner]

    RAG[RAG Pipeline]
    KB[Knowledge Base]
    Embeddings[FastEmbed Embeddings]
    VectorDB[(Qdrant)]

    Tools[Business Tool Registry]
    OrderTool[check_order_status]
    PaymentTool[check_payment_status]
    TicketTool[create_support_ticket]
    EscalationTool[escalate_to_human]

    DB[(PostgreSQL)]
    Memory[Conversation Memory]
    TicketSystem[Support Ticket System]

    LLM[DeepSeek LLM]

    Customer --> Frontend
    Frontend -->|POST /chat| API
    API --> ChatService
    ChatService --> Memory
    ChatService --> Agent

    Agent --> Classifier
    Classifier --> Planner

    Planner --> RAG
    Planner --> Tools

    RAG --> KB
    RAG --> Embeddings
    Embeddings --> VectorDB
    VectorDB --> RAG
    RAG --> LLM

    Tools --> OrderTool
    Tools --> PaymentTool
    Tools --> TicketTool
    Tools --> EscalationTool

    OrderTool --> DB
    PaymentTool --> DB
    TicketTool --> TicketSystem
    EscalationTool --> TicketSystem

    Memory --> DB

    Agent --> LLM
    Agent --> ChatService
    ChatService --> DB
    API --> Frontend
    Frontend --> Customer

7. Application Workflow

Knowledge Query

Customer
   ↓
Streamlit
   ↓
POST /chat
   ↓
FastAPI
   ↓
ChatApplicationService
   ↓
Agent
   ↓
Intent Classification
   ↓
Action Planner
   ↓
RAG
   ↓
Qdrant Retrieval
   ↓
Relevant Context
   ↓
DeepSeek
   ↓
Grounded Response
   ↓
Conversation Memory
   ↓
Streamlit

Order Status

Customer
   ↓
Agent
   ↓
ORDER_STATUS
   ↓
Action Planner
   ↓
check_order_status
   ↓
OrderService
   ↓
PostgreSQL
   ↓
Tool Result
   ↓
Response Generation
   ↓
Customer

Payment Status

Customer
   ↓
Agent
   ↓
PAYMENT_STATUS
   ↓
check_payment_status
   ↓
OrderService + PaymentService
   ↓
PostgreSQL
   ↓
Tool Result
   ↓
Customer Response

Multi-tool Request

Example:

Please check my order 45821 status and tell me whether the payment was successful.

Workflow:

Customer Request
      ↓
Intent Classification
      ↓
Multiple Intents
      ↓
Action Planner
      ↓
┌───────────────────────┐
│ check_order_status    │
│ check_payment_status  │
└───────────────────────┘
      ↓
Execute both tools
      ↓
Combine results
      ↓
Customer response

Human Escalation

Customer
   ↓
HUMAN_ESCALATION
   ↓
escalate_to_human
   ↓
Urgent Support Ticket
   ↓
Customer Confirmation

8. RAG Architecture

The knowledge-base pipeline is:

Knowledge Base Documents
          ↓
Document Loading
          ↓
Normalization
          ↓
Chunking
          ↓
Embedding Generation
          ↓
Qdrant
          ↓
Similarity Retrieval
          ↓
Relevant Context
          ↓
DeepSeek
          ↓
Grounded Answer

Embedding Model

BAAI/bge-small-en-v1.5

Vector dimension:

384

Vector Database

Qdrant
Collection: support_kb
Distance: cosine

Knowledge Base Files

knowledge_base/
├── faq.md
├── product_information.md
├── refund_policy.md
├── cancellation_policy.md
├── shipping_policy.md
├── payment_policy.md
├── account_policy.md
└── support_guidelines.md

Ingestion

Run:

uv run python -m app.ai.rag.ingest --directory knowledge_base

9. Agent Workflow

The agent is implemented using LangGraph.

High-level flow

START
  ↓
Intent Classification
  ↓
Action Planning
  ↓
Workflow Routing
  ├── RAG
  ├── Tool
  ├── Multi-tool
  ├── Mixed RAG + Tool
  ├── Clarification
  ├── Ticket
  └── Escalation
  ↓
Execution
  ↓
Response Generation
  ↓
Finalization
  ↓
END

Design Principle

The LLM is responsible for:

Understanding natural language

Intent classification

Argument extraction

Language generation

The application is responsible for:

Business truth

Database operations

Customer ownership

Input validation

Tool execution

Persistence

Failure handling

This prevents the LLM from becoming the source of truth for transactional information.

10. Intent Model

Supported intents:

KNOWLEDGE_QUERY
ORDER_STATUS
PAYMENT_STATUS
REFUND
CANCELLATION
SUPPORT_TICKET
HUMAN_ESCALATION
UNKNOWN

Multiple intents can be detected in one customer request.

Example:

Check my order 45821 and tell me whether payment succeeded.

Can produce:

ORDER_STATUS
PAYMENT_STATUS

11. Tool Documentation

check_order_status

Purpose:

Retrieve authoritative order information.

Inputs:

order_id
customer_id

Returns information such as:

order_id
status
total_amount
expected_delivery

Customer ownership is validated before returning order information.

check_payment_status

Purpose:

Retrieve payment information associated with an order.

Inputs:

order_id
customer_id

Returns:

order_id
payment_id
transaction_id
status
amount
created_at

create_support_ticket

Purpose:

Create a customer-support ticket.

Typical information:

customer_id
category
description
priority
conversation_id

The ticket is persisted through the application service layer.

escalate_to_human

Purpose:

Escalate a customer issue to human support.

The workflow creates an urgent support ticket associated with the current conversation.

12. Conversation Memory

Conversation memory is persisted in PostgreSQL.

Data model

Customer
   │
   └── Conversation
          │
          └── Messages

A conversation contains:

conversation_id
customer_id
status
created_at
updated_at

Each message contains:

message_id
conversation_id
role
content
created_at

Roles currently persisted for conversational context:

user
assistant

Memory Flow

Incoming customer request
        ↓
Load conversation
        ↓
Load previous messages
        ↓
Agent receives conversation history
        ↓
Generate response
        ↓
Persist user + assistant turns

The backend maintains conversation context across requests.

13. API Documentation

Customer Chat

POST /chat

Request:

{
  "customer_id": "UUID",
  "message": "Where is my order?",
  "conversation_id": null
}

Response:

{
  "conversation_id": "UUID",
  "response": "Please provide your order ID."
}

List Conversations

GET /conversations?customer_id=<UUID>

Returns persisted conversations for the customer.

Get Conversation

GET /conversations/{conversation_id}?customer_id=<UUID>

Returns persisted messages for the selected conversation.

Tickets

POST /tickets
GET /tickets/{id}

Orders

GET /orders/{id}

Payments

GET /payments/{id}

Human Escalation

POST /escalate

The application also contains internal application capabilities through services and tools. The customer-facing Streamlit workflow uses natural-language chat rather than exposing individual business tools as UI actions.

14. Project Structure

customer-support-ai/
│
├── app/
│   ├── ai/
│   │   ├── agent/
│   │   ├── rag/
│   │   └── tools/
│   │
│   ├── api/
│   │   ├── routes/
│   │   └── schemas/
│   │
│   ├── core/
│   ├── db/
│   │   ├── models/
│   │   ├── repositories/
│   │   └── session.py
│   │
│   └── services/
│
├── frontend/
│   ├── __init__.py
│   └── streamlit/
│       ├── __init__.py
│       ├── app.py
│       ├── api_client.py
│       └── config.py
│
├── knowledge_base/
│   ├── faq.md
│   ├── product_information.md
│   ├── refund_policy.md
│   ├── cancellation_policy.md
│   ├── shipping_policy.md
│   ├── payment_policy.md
│   ├── account_policy.md
│   └── support_guidelines.md
│
├── scripts/
│   └── seed_db.py
│
├── tests/
│   ├── evaluation/
│   ├── integration/
│   └── unit/
│
├── alembic/
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── streamlit_app.py
└── README.md

15. Installation

Prerequisites

Install:

Python 3.11+

uv

Docker Desktop

Git

Clone repository

git clone <your-repository-url>
cd customer-support-ai

Install dependencies

uv sync

16. Environment Variables

Create:

.env

Example:

APP_ENV=development
APP_NAME=customer-support-ai

DATABASE_URL=postgresql+psycopg://support_user:support_password@localhost:5432/support_db

DEEPSEEK_API_KEY=<your-key>
LLM_MODEL=deepseek-flash
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=800

QDRANT_URL=<your-qdrant-url>
QDRANT_API_KEY=<your-qdrant-key>
QDRANT_COLLECTION_NAME=support_kb
QDRANT_VECTOR_SIZE=384
QDRANT_TIMEOUT=10.0

BACKEND_API_URL=http://localhost:8000
DEMO_CUSTOMER_ID=<seeded-demo-customer-uuid>

Never commit:

.env

API keys and credentials must not be committed to GitHub.

17. Running the Application

Start PostgreSQL and Qdrant

docker compose up -d

Apply migrations

uv run alembic upgrade head

Seed development data

uv run python -m scripts.seed_db

Ingest knowledge base

uv run python -m app.ai.rag.ingest --directory knowledge_base

Start FastAPI

uv run uvicorn app.main:app --reload

FastAPI:

http://localhost:8000

Swagger:

http://localhost:8000/docs

Start Streamlit

uv run streamlit run streamlit_app.py

Streamlit is available on the local URL displayed by the command.

18. Testing

Run the complete test suite:

uv run pytest -q

Run linting:

uv run ruff check .

Important test scenarios

The test suite covers:

General FAQ

Knowledge-base queries

Refund questions

Order-status requests

Payment-status requests

Support-ticket creation

Human escalation

Unknown questions

Invalid order ID

Tool failure

Retrieval failure

Conversation follow-up

Multiple requests in one conversation

Multi-tool workflows

Mixed RAG + tool workflows

API validation

Conversation persistence

Frontend API failures

19. Error Handling

The application explicitly handles:

Invalid Input

Examples:

Empty customer message

Invalid UUID

Invalid tool arguments

Invalid order ID

Missing Information

The agent asks for the required field instead of executing an incomplete action.

Example:

Customer:
I want to check my order.

Assistant:
Please provide your order ID.

Order Not Found

The system returns a meaningful failure response rather than inventing order information.

Payment Failure

Payment lookup errors are converted into structured tool failures and customer-safe responses.

Retrieval Failure

The RAG workflow records the retrieval error and produces a fallback rather than fabricating knowledge-base content.

LLM Failure

Generation errors are caught and handled through deterministic fallback logic where applicable.

Ticket Failure

Ticket creation errors are propagated as application-level failures and surfaced appropriately.

20. Security and Data Ownership

Business tools validate customer ownership before exposing customer-specific transactional data.

For example:

customer_id + order_id
        ↓
ownership validation
        ↓
database lookup
        ↓
tool result

The application does not trust the LLM to enforce ownership or business authorization.

Credentials are loaded through environment-based configuration.

21. Known Limitations

This project is a development/demo system and does not integrate with real external business systems.

Current limitations include:

No real payment gateway

No real carrier/shipping integration

No production authentication system

No production user-management layer

No real external ticketing platform

Demo customer/order/payment dataset

Local/development infrastructure

No production-grade observability platform

No voice, email, WhatsApp, or SMS channels

The project intentionally keeps the scope focused on the required AI customer-support workflow.

22. Future Improvements

Potential extensions include:

Real CRM integration

Real payment-provider integration

Real shipping-provider integration

Customer authentication

Role-based support-agent dashboard

Conversation archive/restore

Conversation deletion/archive controls

Streaming responses

Better conversation titles

Advanced RAG evaluation

Observability and tracing

Rate limiting

Production deployment

Additional support tools

Human-agent handoff dashboard

23. Sample Customer Queries

Knowledge Base

What is your refund policy?

Can I cancel my order?

How long does shipping take?

Order

Where is my order 45821?

What is the status of order 45821?

Payment

Was my payment successful for order 45821?

Multiple Requests

Check my order 45821 status and tell me whether the payment was successful.

Support Ticket

I want to create a support ticket.

Human Escalation

I want to speak with a human.

Follow-up

My order number is 45821.

Then:

When will it arrive?

24. Demonstration Flow

A complete demonstration should show:

1. Knowledge-base question
        ↓
2. RAG retrieval
        ↓
3. Action-based question
        ↓
4. Tool calling
        ↓
5. Multiple tools
        ↓
6. Conversation memory
        ↓
7. Ticket creation
        ↓
8. Human escalation
        ↓
9. Error scenario
        ↓
10. Architecture explanation

Example multi-turn demo:

Customer:
Where is my order?

Assistant:
Please provide your order ID.

Customer:
45821

Assistant:
Order #45821 has been shipped...

Customer:
When will it arrive?

Assistant:
Uses the existing conversation context to understand
that "it" refers to order 45821.

25. Engineering Principles

Separation of Concerns

Frontend
   ↓
API
   ↓
Application Service
   ↓
Agent
   ↓
RAG / Tools
   ↓
Repositories / Database

LLM vs Application Responsibilities

The LLM handles language understanding and generation.

The application owns:

Transactional truth

Validation

Persistence

Authorization

Business logic

Tool execution

Error handling

Deterministic Business Logic

Transactional information is retrieved from application services and PostgreSQL rather than generated by the LLM.

26. Development Commands

Run tests

uv run pytest -q

Run linting

uv run ruff check .

Start backend

uv run uvicorn app.main:app --reload

Start frontend

uv run streamlit run streamlit_app.py

Start infrastructure

docker compose up -d

Stop infrastructure

docker compose down

27. Project Completion

The implemented system covers:

RAG
  ↓
Agent
  ↓
Tools
  ↓
Conversation Memory
  ↓
FastAPI API
  ↓
Streamlit Frontend
  ↓
Testing
  ↓
Documentation

The architecture follows the project's requirement to implement a meaningful AI support application rather than a basic LLM-only chatbot