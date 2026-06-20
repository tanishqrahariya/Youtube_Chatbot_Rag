import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables (Make sure HUGGINGFACEHUB_API_TOKEN is set)
load_dotenv()

# Set up Streamlit Page Configuration
st.set_page_config(page_title="YouTube Video Q&A", page_icon="📺", layout="wide")
st.title("📺 YouTube Video Q&A Assistant")
st.subheader("Extract knowledge from any YouTube video transcript instantly using RAG")


# 1. Cached functions to optimize performance
@st.cache_data(show_spinner="Fetching and processing YouTube transcript...")
def get_transcript_chunks(video_url_or_id):
    # Extract video ID if full URL is provided
    if "v=" in video_url_or_id:
        video_id = video_url_or_id.split("v=")[1].split("&")[0]
    elif "youtu.be/" in video_url_or_id:
        video_id = video_url_or_id.split("youtu.be/")[1].split("?")[0]
    else:
        video_id = video_url_or_id

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=['en'])
        transcript = " ".join(snippet['text'] if isinstance(snippet, dict) else snippet.text for snippet in transcript_list)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.create_documents([transcript])
        return chunks
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

@st.cache_resource(show_spinner="Initializing Embeddings & Vector Store...")
def create_vector_store(_chunks):
    if not _chunks:
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(_chunks, embeddings)
    return vector_store

@st.cache_resource(show_spinner="Loading Llama 3.1 Model...")
def load_llm_chain():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="conversational",
        max_new_tokens=512,
        temperature=0.5
    )
    model = ChatHuggingFace(llm=llm)
    return model



# 2. Sidebar Setup for Inputs
with st.sidebar:
    st.header("Settings")
    # Prompt user for video link
    video_input = st.text_input("Enter YouTube Video URL or ID:", value="Gfr50f6ZBvo")
    
    # Check if Hugging Face API Token is available
    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        hf_token = st.text_input("Hugging Face API Token:", type="password")
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        else:
            st.warning("Please provide your Hugging Face Token to proceed.")

# 3. Main Logic Execution
if video_input:
    # Initialize components
    chunks = get_transcript_chunks(video_input)
    
    if chunks:
        vector_store = create_vector_store(chunks)
        model = load_llm_chain()
        
        if vector_store and model:
            st.success("🎉 Transcript loaded and indexed successfully! Ask away below.")
            
            # Setup retriever and LCEL chain
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            
            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                Context:
                {context}
                
                Question: {question}
                """,
                input_variables=['context', 'question']
            )

            def format_docs(retrieved_docs):
                return "\n\n".join(doc.page_content for doc in retrieved_docs)

            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })

            parser = StrOutputParser()
            main_chain = parallel_chain | prompt | model | parser

            # 4. Chat User Interface
            st.write("---")
            question = st.text_input("💬 Ask a question about the video:", placeholder="e.g., What is the main topic of the video?")
            
            if question:
                with st.spinner("Analyzing transcript for an answer..."):
                    try:
                        result = main_chain.invoke(question)
                        
                        st.markdown("### 🤖 Answer:")
                        st.info(result)
                    except Exception as e:
                        st.error(f"An error occurred during generation: {e}")