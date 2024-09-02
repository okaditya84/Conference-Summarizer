# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from googletrans import Translator
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# from dotenv import load_dotenv
# from langchain.schema import HumanMessage, SystemMessage

# # Load environment variables
# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Initialize translator
# translator = Translator()

# def translate_text(text, target_language="en"):
#     return translator.translate(text, dest=target_language).text

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             raw_text = page.extract_text()
#             translated_text = translate_text(raw_text)
#             text += translated_text
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. If the answer is not available in the context, say, "answer is not available in the context." Do not provide incorrect information.\n\n
#     Context:\n{context}?\n
#     Question:\n{question}\n
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # Enable dangerous deserialization
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     st.write("Reply:", response["output_text"])

# def summarize_text_with_chat_model(text):
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     messages = [HumanMessage(content=f"Please provide a concise summary of the following text:\n\n{text}")]
#     response = model.invoke(messages)
#     return response.content  # Access the content directly

# def sentiment_analysis_with_chat_model(text):
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     messages = [HumanMessage(content=f"Analyze the sentiment of the following text:\n\n{text}")]
#     response = model.invoke(messages)
#     return response.content  # Access the content directly

# def generate_word_cloud(text):
#     wordcloud = WordCloud(background_color='white', max_words=100).generate(text)
#     plt.figure(figsize=(10, 8))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     st.pyplot(plt)

# def topic_detection(text_chunks):
#     topics = ["Budget Planning", "Project Roadmap", "Resource Allocation", "Team Performance"]
#     st.write("Detected Topics:")
#     for topic in topics:
#         st.write(f"- {topic}")
#     st.write("You might want to ask about these topics!")

# def main():
#     st.set_page_config(page_title="Conference Summarizer")
#     st.header("Chat with your personal assistant 💁")

#     if "raw_text" not in st.session_state:
#         st.session_state.raw_text = ""

#     user_question = st.text_input("Ask a question based on your office meetings, conferences, and more.")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your meeting documents and resources, then click 'Submit & Process'", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             if pdf_docs:
#                 with st.spinner("Processing..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     st.session_state.raw_text = raw_text  # Save raw_text in session state
#                     text_chunks = get_text_chunks(raw_text)
#                     get_vector_store(text_chunks)
#                     st.success("Processing Complete")
#             else:
#                 st.warning("Please upload at least one document.")

#         if st.button("Generate Summary"):
#             if st.session_state.raw_text:
#                 with st.spinner("Generating Summary..."):
#                     summary = summarize_text_with_chat_model(st.session_state.raw_text)
#                     st.write("Summary of Documents:")
#                     st.write(summary)
#             else:
#                 st.warning("Please upload and process the documents first.")

#         if st.session_state.raw_text:
#             st.write("Sentiment Analysis:")
#             sentiment = sentiment_analysis_with_chat_model(st.session_state.raw_text)
#             st.write(sentiment)

#             st.write("Visual Representation of Key Topics:")
#             generate_word_cloud(st.session_state.raw_text)

#             topic_detection(st.session_state.raw_text)

# if __name__ == "__main__":
#     main()




import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from googletrans import Translator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# Initialize translator
translator = Translator()

def translate_text(text, target_language="en"):
    try:
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        st.error(f"Translation Error: {e}")
        return text  # Return original text if translation fails

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text = page.extract_text()
            if raw_text:
                translated_text = translate_text(raw_text)
                text += translated_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store

def generate_response(question, chat_history):
    vector_store = load_vector_store()
    relevant_docs = vector_store.similarity_search(question, k=3)

    system_prompt = SystemMessage(content="You are a helpful and knowledgeable meeting assistant. Provide clear and concise answers based on the provided context.")

    # Build conversation history
    messages = [system_prompt]
    for user_msg, assistant_msg in chat_history:
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=assistant_msg))

    # Add user's latest question
    messages.append(HumanMessage(content=question))

    # Define prompt template
    prompt_template = PromptTemplate(
        template="""
Use the following meeting documents to answer the question. If the answer is not found, respond with "I'm sorry, I could not find the information in the provided documents."

Context:
{context}

Question:
{question}

Answer:""",
        input_variables=["context", "question"]
    )

    # Format context from relevant documents
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Initialize model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

    # Generate response
    prompt = prompt_template.format(context=context, question=question)
    messages.append(SystemMessage(content=prompt))
    response = model.invoke(messages)

    return response.content.strip()

def summarize_text(text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = f"Provide a concise and comprehensive summary of the following meeting documents:\n\n{text}\n\nSummary:"
    messages = [SystemMessage(content=prompt)]
    response = model.invoke(messages)
    return response.content.strip()

def sentiment_analysis(text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = f"Analyze the overall sentiment (positive, neutral, negative) of the following meeting documents and provide a brief explanation:\n\n{text}\n\nSentiment Analysis:"
    messages = [SystemMessage(content=prompt)]
    response = model.invoke(messages)
    return response.content.strip()

def generate_word_cloud(text):
    wordcloud = WordCloud(background_color='white', width=800, height=400).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def topic_detection(text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = f"Identify and list the main topics discussed in the following meeting documents:\n\n{text}\n\nTopics:"
    messages = [SystemMessage(content=prompt)]
    response = model.invoke(messages)
    topics = response.content.strip().split('\n')
    return topics

def main():
    st.set_page_config(page_title="Conference Summarizer", layout="wide")
    st.title("📄 Conference Summarizer and Chat Assistant 💬")

    # Sidebar for uploading and processing documents
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your meeting documents (PDF format):", accept_multiple_files=True, type=['pdf']
        )

        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(uploaded_files)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.session_state['raw_text'] = raw_text
                        st.success("Documents processed successfully!")
                    else:
                        st.error("No text found in the uploaded documents.")
            else:
                st.warning("Please upload at least one PDF document.")

        # Display additional analysis options if documents are processed
        if 'raw_text' in st.session_state:
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = summarize_text(st.session_state['raw_text'])
                    st.subheader("Summary:")
                    st.write(summary)

            if st.button("Sentiment Analysis"):
                with st.spinner("Analyzing sentiment..."):
                    sentiment = sentiment_analysis(st.session_state['raw_text'])
                    st.subheader("Sentiment Analysis:")
                    st.write(sentiment)

            if st.button("Generate Word Cloud"):
                with st.spinner("Generating word cloud..."):
                    generate_word_cloud(st.session_state['raw_text'])
            
            if st.button("Detect Topics"):
                with st.spinner("Detecting topics..."):
                    topics = topic_detection(st.session_state['raw_text'])
                    st.subheader("Detected Topics:")
                    for topic in topics:
                        st.write(f"- {topic}")

    # Chat interface
    st.header("Chat with Your Meeting Assistant")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    user_question = st.text_input("Enter your question:", key="user_input")
    if st.button("Send", key="send_button"):
        if 'raw_text' in st.session_state:
            if user_question:
                with st.spinner("Generating response..."):
                    chat_history = st.session_state['chat_history']
                    response = generate_response(user_question, chat_history)
                    chat_history.append((user_question, response))
                    st.session_state['chat_history'] = chat_history
            else:
                st.warning("Please enter a question.")
        else:
            st.warning("Please upload and process meeting documents first.")

    # Display chat history
    if 'chat_history' in st.session_state and st.session_state['chat_history']:
        st.subheader("Conversation History:")
        for i, (question, answer) in enumerate(st.session_state['chat_history']):
            st.markdown(f"**You:** {question}")
            st.markdown(f"**Assistant:** {answer}")

    # Clear chat history button
    if st.button("Clear Conversation"):
        st.session_state['chat_history'] = []
        st.success("Conversation history cleared.")

if __name__ == "__main__":
    main()
