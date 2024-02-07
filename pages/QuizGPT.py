import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


st.set_page_config(page_title="QuizGPT", page_icon="❓")

st.markdown(
    """
    ## QuizGPT

    ---
    """
)


llm = ChatOpenAI(
    temperature=0.5,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.

    Based only on the following context make 10 questions to test the user's knowledge about the text.

    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    Use (O) to signal the correct answer.

    Question examples:

    Question: What is the color of the ocean?
    Answers: Red | Yellow | Green | Blue(O)

    Question: What is the capital or Georgia?
    Answers: Baku | Tbilisi(O) | Manila | Beirut

    Question: When was Avatar released?
    Answers: 2007 | 2001 | 2009(O) | 1998

    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(O) | Painter | Actor | Model

    Your turn!

    Context: {context}
""",
        )
    ]
)

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a powerful formatting algorithm.

You format exam questions into JSON format.
Answers with (O) are the correct ones.

Example Input:
Question: What is the color of the ocean?
Answers: Red|Yellow|Green|Blue(O)

Question: What is the capital or Georgia?
Answers: Baku|Tbilisi(O)|Manila|Beirut

Question: When was Avatar released?
Answers: 2007|2001|2009(O)|1998

Question: Who was Julius Caesar?
Answers: A Roman Emperor(O)|Painter|Actor|Model


Example Output:

```json
{{ "questions": [
        {{
            "question": "What is the color of the ocean?",
            "answers": [
                    {{
                        "answer": "Red",
                        "correct": false
                    }},
                    {{
                        "answer": "Yellow",
                        "correct": false
                    }},
                    {{
                        "answer": "Green",
                        "correct": false
                    }},
                    {{
                        "answer": "Blue",
                        "correct": true
                    }},
            ]
        }},
                    {{
            "question": "What is the capital or Georgia?",
            "answers": [
                    {{
                        "answer": "Baku",
                        "correct": false
                    }},
                    {{
                        "answer": "Tbilisi",
                        "correct": true
                    }},
                    {{
                        "answer": "Manila",
                        "correct": false
                    }},
                    {{
                        "answer": "Beirut",
                        "correct": false
                    }},
            ]
        }},
                    {{
            "question": "When was Avatar released?",
            "answers": [
                    {{
                        "answer": "2007",
                        "correct": false
                    }},
                    {{
                        "answer": "2001",
                        "correct": false
                    }},
                    {{
                        "answer": "2009",
                        "correct": true
                    }},
                    {{
                        "answer": "1998",
                        "correct": false
                    }},
            ]
        }},
        {{
            "question": "Who was Julius Caesar?",
            "answers": [
                    {{
                        "answer": "A Roman Emperor",
                        "correct": true
                    }},
                    {{
                        "answer": "Painter",
                        "correct": false
                    }},
                    {{
                        "answer": "Actor",
                        "correct": false
                    }},
                    {{
                        "answer": "Model",
                        "correct": false
                    }},
            ]
        }}
    ]
 }}
```
Your turn!

Questions: {context}
""",
        )
    ]
)

formatting_chain = formatting_prompt | llm


@st.cache_resource(show_spinner="Loading...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",
        chunk_size=512,
        chunk_overlap=128,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


with st.sidebar:
    docs = None
    choice = st.selectbox("Choose one", ["File", "Wikipedia"])
    if choice == "File":
        file = st.file_uploader("Upload a file", type=["txt", "docx", "pdf"])
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia for a topic")
        if topic:
            retriever = WikipediaRetriever(top_k_results=5, lang="en")
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)


if not docs:
    st.markdown(
        """
    ### Welcome to QuizGPT

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    start = st.button("Start")

    if start:
        questions_response = questions_chain.invoke(docs)
        st.write(questions_response.content)
        formatting_response = formatting_chain.invoke(
            {"context": questions_response.content}
        )
        st.write(formatting_response.content)
