# W5D2 Model Comparison

## Models Compared

- `llama3.2:3b`
- `qwen2.5:3b`

Both models were tested locally through Ollama using the same system prompt and the same three questions.

## System Prompt

The models were instructed to act as an AI/ML mentor for engineering students who are beginners.

The system prompt asked the models to:

- Explain concepts clearly and simply.
- Use practical examples when useful.
- Avoid unnecessary jargon.
- Break difficult topics into smaller steps.

## Questions Tested

1. What is machine learning?
2. Explain supervised learning with a simple example.
3. What is overfitting and how can it be reduced?

## Observed Differences

### 1. Machine Learning

**Llama 3.2:3b**

- Gave a structured explanation of machine learning.
- Used a child learning to recognize animals as an analogy.
- Explained supervised, unsupervised, and reinforcement learning.
- Included several application areas.

**Qwen 2.5:3b**

- Gave a concise definition of machine learning.
- Used customer purchase history as a practical example.
- Focused more on real-world applications such as recommendations, fraud detection, and predictive maintenance.

**Observation:** Llama 3.2 provided a broader conceptual explanation, while Qwen 2.5 focused more on a practical business example and applications.

### 2. Supervised Learning

**Llama 3.2:3b**

- Used handwritten digit recognition as the example.
- Explained data collection, preparation, training, and evaluation.
- Connected the example to classification.

**Qwen 2.5:3b**

- Used student study time and test scores as the example.
- Clearly identified features and labels.
- Explained the process using a simple linear regression example.

**Observation:** Llama 3.2 gave a more detailed machine-learning workflow, while Qwen 2.5 used a simpler everyday example and explicitly explained features and labels.

### 3. Overfitting

**Llama 3.2:3b**

- Explained overfitting using house-price prediction.
- Described the difference between learning useful patterns and learning noise.
- Suggested regularization, dropout, early stopping, data augmentation, ensemble methods, and simpler models.

**Qwen 2.5:3b**

- Explained overfitting using image recognition.
- Discussed causes such as insufficient data, high model complexity, and data leakage.
- Suggested cross-validation, regularization, early stopping, data augmentation, feature selection, ensemble methods, and dimensionality reduction.

**Observation:** Both models explained overfitting correctly. Llama 3.2 emphasized practical techniques, while Qwen 2.5 provided a broader discussion of causes and prevention strategies.

## Overall Comparison

| Aspect                | Llama 3.2:3b                              | Qwen 2.5:3b                                           |
| --------------------- | ----------------------------------------- | ----------------------------------------------------- |
| Explanation style     | Detailed and structured                   | Concise and practical                                 |
| Examples              | Animals, handwritten digits, house prices | Customer purchases, student scores, image recognition |
| Beginner friendliness | High                                      | High                                                  |
| Technical depth       | Higher in some responses                  | Broad coverage                                        |
| Practical examples    | Good                                      | Strong                                                |
| Response length       | Generally longer                          | Generally more concise                                |

## Conclusion

Both local models successfully answered the same three AI/ML questions using the same system prompt.

For beginner-oriented learning, both models produced useful explanations. Llama 3.2:3b generally provided more detailed and structured responses, while Qwen 2.5:3b tended to provide concise explanations with practical examples.

The comparison demonstrates that different local LLMs can produce different response styles even when they receive the same prompts and system instructions.

## Comparison Method

The comparison was performed locally using Ollama.

Both models received:

- The same system prompt
- The same three questions
- The same Python API implementation

The observations in this document are based on the actual generated responses saved in `outputs/model_comparison.txt`.
