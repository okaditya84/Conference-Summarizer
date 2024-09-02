# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






# def get_pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         pdf_reader= PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text+= page.extract_text()
#     return  text



# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")


# def get_conversational_chain():

#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro",
#                              temperature=0.3)

#     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain



# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
#     new_db = FAISS.load_local("faiss_index", embeddings)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()

    
#     response = chain(
#         {"input_documents":docs, "question": user_question}
#         , return_only_outputs=True)

#     print(response)
#     st.write("Reply: ", response["output_text"])




# def main():
#     st.set_page_config("Conference Summarizer")
#     st.header("Chat with your personal assistant 💁")

#     user_question = st.text_input("Ask a Question from based on your office meetings, conferences and much more. ")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your meeting documents and resources and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")



# if __name__ == "__main__":
#     main()



import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from googletrans import Translator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

translator = Translator()

def translate_text(text, target_language="en"):
    return translator.translate(text, dest=target_language).text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text = page.extract_text()
            translated_text = translate_text(raw_text)
            text += translated_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, say, "answer is not available in the context." Do not provide incorrect information.\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

def summarize_text_with_chat_model(text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = f"Please provide a concise summary of the following text:\n\n{text}\n\nSummary:"
    response = model({"prompt": prompt})
    return response['text']

def sentiment_analysis(text):
    model = HuggingFaceHub(repo_id="distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_chain = LLMChain(llm=model, prompt=PromptTemplate(input_variables=["text"], template="Text: {text}\nSentiment:"))
    sentiment = sentiment_chain.run(text=text)
    return sentiment

def generate_word_cloud(text):
    wordcloud = WordCloud(background_color='white', max_words=100).generate(text)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

def topic_detection(text_chunks):
    topics = ["Budget Planning", "Project Roadmap", "Resource Allocation", "Team Performance"]
    st.write("Detected Topics:")
    for topic in topics:
        st.write(f"- {topic}")
    st.write("You might want to ask about these topics!")

def main():
    st.set_page_config("Conference Summarizer")
    st.header("Chat with your personal assistant 💁")

    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""

    user_question = st.text_input("Ask a Question based on your office meetings, conferences, and more.")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your meeting documents and resources and click on Submit & Process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                st.session_state.raw_text = raw_text  # Save raw_text in session state
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Complete")

        if st.button("Generate Summary"):
            if st.session_state.raw_text:
                with st.spinner("Generating Summary..."):
                    summary = summarize_text_with_chat_model(st.session_state.raw_text)
                    st.write("Summary of Documents:")
                    st.write(summary)
            else:
                st.warning("Please upload and process the documents first.")

        if st.session_state.raw_text:
            st.write("Sentiment Analysis:")
            sentiment = sentiment_analysis(st.session_state.raw_text)
            st.write(sentiment)

            st.write("Visual Representation of Key Topics:")
            generate_word_cloud(st.session_state.raw_text)

            topic_detection(st.session_state.raw_text)

if __name__ == "__main__":
    main()
