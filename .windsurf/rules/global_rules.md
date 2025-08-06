---
trigger: always_on
description: 
globs: 
---

1. Only use pytest and pytest-mock for testing. Do not use unittest or any other testing frameworks.

2. If fixing or updating a test requires changing the implementation code:
   - Clearly explain what the issue is and why the test is failing.
   - Ask for confirmation before proceeding with the implementation changes.

3. Always explain your reasoning before writing code.
   - Provide a brief step-by-step plan or bullet-point summary.
   - Make it clear how each step addresses the problem.

4. Avoid assumptions about the developer’s intent. If unclear:
   - Ask clarifying questions before taking action.
   - Do not generate code until intent is confirmed.

5. Code must be self-explanatory and readable:
   - Use meaningful variable and function names.
   - Include a one-line comment above complex logic blocks.

6. Group related changes together.
   - Do not mix unrelated fixes or refactors in a single update.
   - State explicitly what category each change belongs to (e.g., bugfix, test improvement, refactor).

7. Never delete code unless explicitly instructed to do so.

8. If you propose a refactor:
   - Justify the benefit (e.g., clarity, performance, testability).
   - Ask for approval before applying large structural changes.

9. Treat the codebase as if it's in production.
   - No experimental or incomplete code unless requested.
   - Ensure code passes linting and all existing tests.

10. Format output like ChatGPT:
    - Use Markdown headings (`###`), code blocks (```python), and bullet points for clarity.
    - Avoid internal language model meta-talk (e.g., “As an AI...”).