SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Think step by step, then output ONLY the final answer "
    "in the required format."
)

CSQA_USER_TEMPLATE = (
    "Answer the multiple-choice question.\n\n"
    "Question:\n{question}\n\n"
    "Choices:\n{choices}\n\n"
    "Return ONLY: Final Answer: <A|B|C|D|E>\n"
)

STRATEGYQA_USER_TEMPLATE = (
    "Answer the yes/no question.\n\n"
    "Question:\n{question}\n\n"
    "Return ONLY: Final Answer: <yes|no>\n"
)
