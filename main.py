from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()



embeddings = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k":10,
        "lambda_mult" :0.5
    }
)


llm = ChatMistralAI(model="mistral-small-2603")

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("Rag system created ")
print("ask anything about deep learning... ")
print("Press 0 to exit")

while True:
    query = input("You : ")
    if query == 0:
        break

    docs = retriever.invoke(query)

    print(docs)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    print("-----------------context-----------")
    print(context)
    print("-----------------context end-----------")

    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")