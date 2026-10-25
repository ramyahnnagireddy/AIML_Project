# W9D4 — LangChain Hybrid Agent Systems

- Model: `qwen2.5:3b`
- Chain: `PromptTemplate → OllamaLLM → StrOutputParser`
- Memory: `ConversationBufferMemory`
- Tools: `web_search_stub`, `calculator`

## 1. Chain — five inputs

### Input 1
**Question:** Explain RAG in one sentence.

**Output:** RAG stands for Retrieval-Augmented Generation, which enhances text generation by using pre-existing documents to inform and improve the output.

### Input 2
**Question:** Explain why MLflow is useful in machine learning.

**Output:** MLflow is useful in machine learning because it provides a platform for tracking experiments, managing models, and deploying machine learning models, making it easier to manage and reproduce machine learning workflows.

### Input 3
**Question:** Give one benefit of AI in healthcare.

**Output:** One benefit of AI in healthcare is its ability to analyze large volumes of medical data quickly, enabling faster and more accurate diagnoses and treatment recommendations.

### Input 4
**Question:** Explain one basic cybersecurity practice.

**Output:** One basic cybersecurity practice is to regularly update software and operating systems to protect against known vulnerabilities and security flaws.

### Input 5
**Question:** Explain what LangChain helps developers build.

**Output:** LangChain helps developers build more sophisticated and modular applications by providing tools and libraries to integrate natural language processing and AI capabilities into their projects.

## 2. Conversation memory — five turns

### Turn 1
- User: My project uses RAG.
- Assistant: RAG retrieves relevant context for a language model.
- History characters after turn: 84

### Turn 2
- User: Which tool tracks experiments?
- Assistant: MLflow can track machine-learning experiments.
- History characters after turn: 173

### Turn 3
- User: What model am I using today?
- Assistant: You are using the local qwen2.5:3b model.
- History characters after turn: 255

### Turn 4
- User: Why keep conversation history?
- Assistant: History helps preserve context across turns.
- History characters after turn: 342

### Turn 5
- User: Summarize my project focus.
- Assistant: Your focus is a LangChain chain, memory, and tools.
- History characters after turn: 433

**Verification:** All five user messages and assistant responses were found in the final conversation buffer.

## 3. Tool-using agent — three tasks

### Task 1
- Task: Search for an explanation of RAG.
- Selected tool: `web_search_stub`
- Result: Offline search result for 'an explanation of RAG': Retrieval-Augmented Generation (RAG) retrieves relevant documents and provides their context to a language model before it answers.

### Task 2
- Task: Calculate 18 * 24.
- Selected tool: `calculator`
- Result: 18 * 24 = 432

### Task 3
- Task: Search for the purpose of MLflow.
- Selected tool: `web_search_stub`
- Result: Offline search result for 'the purpose of MLflow': MLflow supports experiment tracking, model evaluation, and management of machine-learning model lifecycles.

## Final memory buffer

```text
Human: My project uses RAG.
AI: RAG retrieves relevant context for a language model.
Human: Which tool tracks experiments?
AI: MLflow can track machine-learning experiments.
Human: What model am I using today?
AI: You are using the local qwen2.5:3b model.
Human: Why keep conversation history?
AI: History helps preserve context across turns.
Human: Summarize my project focus.
AI: Your focus is a LangChain chain, memory, and tools.
```

## Notes

- The search tool is a predefined offline stub, not live web search.
- The agent uses an explicit routing policy to select tools.
- LLM responses are generated locally using Ollama.
