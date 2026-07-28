"""结构化摘要使用的提示词定义。"""

from langchain_core.prompts import ChatPromptTemplate

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarize only the supplied text. Preserve its facts and do not invent "
            "information or claim external research. Use the requested language when one "
            "is provided. Keep the summary within the requested maximum word count. Return "
            "a summary and concise key_points. Requested language: {language}. Maximum "
            "words: {max_words}.",
        ),
        ("human", "Text to summarize:\n{text}"),
    ]
)
