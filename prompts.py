from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
Trả lời câu hỏi dựa trên ngữ cảnh sau đây:

{context}

Tìm kiếm tương tự hàng đầu: {question}
"""
def create_prompt(context, question):

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(context=context, question=question)
    return prompt