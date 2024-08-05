from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

app = Flask(__name__)
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_project = os.getenv("LANGCHAIN_PROJECT")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")

llm = ChatOpenAI(model="gpt-3.5-turbo")

markdown_path = "documentation.md"
loader = UnstructuredMarkdownLoader(markdown_path)
data = loader.load()
assert len(data) == 1
assert isinstance(data[0], Document)
readme_content = data[0].page_content

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500, add_start_index=True)
all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

system_prompt = (
    "You are an assistant for question-answering tasks related to game development using a custom game engine called Terra Creator Studio. "
    "Your primary responsibilities include: "
    "- Providing complete function implementations in T# when a script or code snippet is requested. "
    "- Ensure the function code is self-contained and can be directly copied into Terra Studio."
    "- Prioritize using or modifying existing wrapper functions for logic templates, or combine multiple wrapper functions, rather than writing new functions from scratch unless no suitable options exist. "
    "Follow these guidelines: "
    "- Most syntax is similar to C#, but identify differences by examining access wrappers and methods. "
    "- Refer to the 'T# Don'ts' section of the context document to avoid common pitfalls and differences from Unity C# syntax. "
    "- If a question requests a script, provide complete T# function code that can be copied directly into Terra Studio. "
    "Consistency is key: "
    "- Ensure responses are consistent; similar questions should yield similar answers, even if asked multiple times. "
    "- Always refer to the provided context below. "
    "If you encounter questions about features in Terra Studio: "
    "- Search the relevant portion of the context and provide a relevant answer. "
    "Important: "
    "- Always double-check the context document to ensure your answers are accurate and not based on hallucination. "
    "- Verifying information is more important than providing a quick but incorrect answer. "
    "If you cannot find an answer in the T# documentation, state that you don't know. "
    "Your answers should be: "
    "- Clear, concise, and suitable for novice developers. "
    "- Always include the source of the information used in your response from the context."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    session_id = data.get('session_id', 'default_session')
    input_question = data.get('input')
    retrieved_docs = retriever.invoke(input_question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    response = conversational_rag_chain.invoke(
        {"input": input_question, "context": context},
        config={"configurable": {"session_id": session_id}}
    )
    return jsonify({"answer": response["answer"]})

if __name__ == '__main__':
    app.run(debug=True)