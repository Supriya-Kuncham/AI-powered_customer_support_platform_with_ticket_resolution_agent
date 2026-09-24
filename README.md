# SupportPilot — AI-Powered Customer Support Platform

**SupportPilot** is an AI-powered customer support platform designed to automate the lifecycle of IT support tickets—from ticket classification and knowledge retrieval to resolution validation and human escalation.

The system combines **Machine Learning, Retrieval-Augmented Generation (RAG), and a multi-agent architecture** to provide evidence-based resolutions while escalating uncertain cases to human support.

---

## Overview

Modern support teams handle a large volume of repetitive IT-related issues such as:

* VPN and network connectivity problems
* Password and account issues
* Software and system errors
* Hardware-related problems
* Access and configuration requests

Manually analyzing and resolving every ticket can be time-consuming.

SupportPilot addresses this by providing an automated workflow that:

1. Understands the incoming support ticket
2. Classifies the ticket based on its characteristics
3. Retrieves relevant troubleshooting information
4. Generates a resolution based on the retrieved knowledge
5. Validates the quality and confidence of the proposed resolution
6. Automatically resolves sufficiently confident tickets
7. Escalates uncertain tickets to human support

---

# System Architecture

```text
                         Customer
                            │
                            ▼
                     Support Ticket
                            │
                            ▼
                  ┌───────────────────┐
                  │  Diagnosis Agent  │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Retrieval Agent   │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Resolution Agent  │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Validation Agent  │
                  └─────────┬─────────┘
                            │
                    ┌───────┴────────┐
                    │                │
                    ▼                ▼
              Auto Resolve       Escalation
                                     │
                                     ▼
                           ┌─────────────────┐
                           │ Escalation Agent│
                           └────────┬────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                    Jira Ticket          Email Notification
```

The workflow is coordinated by the **SupportPilot Orchestrator**, which manages the interaction between the specialized agents.

---

# Key Features

### 1. Intelligent Ticket Classification

SupportPilot uses a machine-learning pipeline to analyze incoming support tickets.

The classification pipeline uses:

* **TF-IDF** for text feature extraction
* **Logistic Regression** for classification

The system predicts relevant ticket attributes such as:

* Category
* Severity
* Priority

The initial classification model achieved approximately **68% accuracy** on the evaluated dataset.

The result was retained as measured rather than artificially optimized, since overlapping support categories can make classification difficult.

---

### 2. Retrieval-Augmented Generation (RAG)

SupportPilot uses **Retrieval-Augmented Generation** to ground troubleshooting responses in a knowledge base.

Instead of relying only on generated content, the system:

```text
Ticket
  ↓
Retrieve relevant knowledge
  ↓
Identify supporting information
  ↓
Generate resolution
  ↓
Return response with source
```

The retrieval pipeline uses **similarity-based matching** to identify relevant troubleshooting information.

This allows the generated resolution to be associated with the source information used during the response generation process.

---

### 3. Multi-Agent Architecture

SupportPilot divides the support workflow into specialized agents.

#### Diagnosis Agent

Analyzes the ticket and determines the likely:

* Category
* Severity
* Priority
* Problem type

#### Retrieval Agent

Searches the knowledge base and identifies information relevant to the diagnosed issue.

#### Resolution Agent

Uses the retrieved information to generate actionable troubleshooting steps.

#### Validation Agent

Evaluates the proposed resolution before it is returned to the customer.

The current validation mechanism combines:

```text
40% — Diagnosis confidence
40% — Retrieval confidence
20% — Response completeness
```

The resulting score is used to determine whether the ticket can proceed through automatic resolution or should be escalated.

#### Escalation Agent

Handles tickets that do not meet the configured confidence requirements.

It can preserve the available ticket context and initiate external support actions such as Jira ticket creation and email notification.

---

# Human-in-the-Loop Design

A key design principle of SupportPilot is:

> **When the system does not have sufficient confidence, it should escalate instead of producing an unsupported resolution.**

```text
                AI Resolution
                     │
                     ▼
                 Validation
                     │
             ┌───────┴───────┐
             │               │
        High Confidence   Low Confidence
             │               │
             ▼               ▼
        Auto Resolve       Escalate
                             │
                     ┌───────┴──────┐
                     ▼              ▼
                   Jira           Email
```

This creates a human-in-the-loop support workflow rather than relying on automation for every ticket.

---

# Authentication & Security

SupportPilot includes authentication and security mechanisms for application access.

### Password Hashing

User passwords are hashed before storage rather than being stored as plain text.

### JWT Authentication

The application uses **JSON Web Tokens (JWT)** to authenticate users and maintain authenticated sessions.

### Google OAuth

Google OAuth authentication is supported for simplified user sign-in.

### Environment-Based Secrets

Sensitive credentials are maintained outside the application source code through environment configuration.

Examples include:

* JWT secrets
* API keys
* OAuth credentials
* External service credentials

These values should not be committed to the public repository.

---

# External Integrations

## Jira

SupportPilot can integrate with Jira to create support issues when a ticket requires human intervention.

This allows an AI-generated workflow to transition into an existing enterprise support process.

## Email

Email integration allows the system to notify relevant users when support actions are completed or escalation is required.

---

# Technology Stack

| Technology          | Purpose                       |
| ------------------- | ----------------------------- |
| Python              | Core application and AI logic |
| Flask               | Web application framework     |
| Scikit-learn        | Machine Learning              |
| TF-IDF              | Text feature extraction       |
| Logistic Regression | Ticket classification         |
| RAG                 | Knowledge retrieval           |
| Cosine Similarity   | Similarity-based retrieval    |
| JWT                 | Authentication                |
| OAuth 2.0           | Google authentication         |
| SQLite              | Data persistence              |
| Jira API            | Support-ticket escalation     |
| Email Service       | Notifications                 |
| HTML/CSS            | User interface                |
| Git/GitHub          | Version control               |

---

# Project Structure

```text
SupportPilot/
│
├── app.py
├── agents.py
├── auth_jwt.py
├── classifier.py
├── database.py
├── email_service.py
├── jira_service.py
├── knowledge_base.py
├── rag_pipeline.py
├── train_model.py
├── evaluate_retrieval.py
│
├── data/
│   └── IT_Support_Ticket_Data.csv
│
├── models/
│   ├── category_model.pkl
│   ├── category_vectorizer.pkl
│   ├── severity_model.pkl
│   ├── severity_vectorizer.pkl
│   └── ticket_classifier.pkl
│
├── templates/
│   ├── index.html
│   ├── login.html
│   └── register.html
│
├── static/
│   └── logo.jpeg
│
├── evaluation_report.json
├── rag_evaluation_report.json
├── requirements.txt
└── README.md
```

---

# Installation

## Prerequisites

* Python 3.x
* Git
* Required API credentials for optional external integrations

## 1. Clone the Repository

```bash
git clone https://github.com/Supriya-Kuncham/AI-powered_customer_support_platform_with_ticket_resolution_agent.git
```

```bash
cd AI-powered_customer_support_platform_with_ticket_resolution_agent
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create the required environment configuration and add the credentials for the services being used.

Do not commit secrets, API keys, OAuth credentials, or private configuration files to the repository.

## 5. Run the Application

```bash
python app.py
```

The application will start using the Flask development server.

---

# Application Workflow

A typical SupportPilot interaction follows this process:

```text
User Login
    ↓
Submit Support Ticket
    ↓
Ticket Classification
    ↓
Diagnosis
    ↓
Knowledge Retrieval
    ↓
Resolution Generation
    ↓
Validation
    ↓
┌───────────────────────┐
│                       │
▼                       ▼
Auto Resolution       Escalation
                         │
                  ┌──────┴──────┐
                  ▼             ▼
                Jira          Email
```

---

# Evaluation

The project includes evaluation artifacts for both machine-learning classification and retrieval components.

The classification model achieved approximately **68% accuracy** during evaluation.

The RAG pipeline was evaluated using retrieval-specific evaluation data and reports generated during development.

The multi-agent workflow was also tested using support-ticket scenarios covering both:

* **AUTO_RESOLVE**
* **ESCALATE**

outcomes.

---

# Project Development Milestones

## Milestone 1 — Ticket Classification

Implemented the initial machine-learning pipeline for understanding support tickets.

**Key components:**

* Dataset preparation
* TF-IDF vectorization
* Logistic Regression
* Category classification
* Severity prediction
* Priority handling
* Model evaluation

---

## Milestone 2 — RAG Knowledge Retrieval

Extended the system beyond classification by introducing knowledge-based troubleshooting.

**Key components:**

* Knowledge-base creation
* Retrieval pipeline
* Similarity-based matching
* Source-aware responses
* Retrieval evaluation

---

## Milestone 3 — Multi-Agent Support Automation

Extended the RAG workflow into a coordinated multi-agent system.

**Key components:**

* Diagnosis Agent
* Retrieval Agent
* Resolution Agent
* Validation Agent
* Escalation Agent
* SupportPilot Orchestrator
* Confidence-based routing
* Jira integration
* Email integration
* End-to-end workflow testing

---

# Future Enhancements

Potential improvements include:

* Improve classification performance with additional and higher-quality training data
* Introduce embedding-based semantic retrieval
* Expand the troubleshooting knowledge base
* Add additional enterprise integrations
* Improve multilingual ticket handling
* Add conversation history
* Implement advanced monitoring and analytics
* Expand automated evaluation of generated resolutions
* Improve model and retrieval explainability

---

# Project Outcome

SupportPilot demonstrates an end-to-end approach to AI-assisted customer support by combining:

**Machine Learning → RAG → Multi-Agent Orchestration → Validation → Human Escalation → Enterprise Integration**

Rather than treating AI as only a text-generation tool, the project focuses on building a structured support workflow where the system can **understand, retrieve, resolve, validate, and escalate** support tickets.

---


