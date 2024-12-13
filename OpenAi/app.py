import streamlit as st
import fitz  # PyMuPDF
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv

load_dotenv()

# Function to display logo at the top of the app
def display_logo():
    # Update this path to the location of your logo
    logo_path = "NajmAI-Logo.png"  # Adjust path as necessary
    st.image(logo_path, width=350)  # You can change the width or use 'use_column_width=True'

# Function to extract text from doc2.pdf
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to split the text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Adjust model as needed
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# Function to create RAG chain
def create_rag_chain(vector_store):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 1})  # Retrieve top 1 relevant chunk
    )
    return qa_chain

#--------------------------------------------------------------------

from langchain.prompts import PromptTemplate

def create_rag_chain_with_prompt(vector_store, driver1_name, driver1_car_plate, driver1_accident_desc, driver2_name, driver2_car_plate, driver2_accident_desc):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})  # Retrieve top 1 relevant chunk
    
    # Define the simplified prompt to only ask for the final decision
    
    prompt_template = """

        You are an expert in traffic law and accident case evaluation. Based on the details of the case provided below, please determine who is responsible for the accident using general traffic law principles (e.g., right of way, speed limits, reckless driving, lane discipline, etc.).

        **Case Details:**
        - **Driver 1 Information:**
        - Name: {driver1_name}
        - Car Plate: {driver1_car_plate}
        - Accident Description: {driver1_accident_desc}

        - **Driver 2 Information:**
        - Name: {driver2_name}
        - Car Plate: {driver2_car_plate}
        - Accident Description: {driver2_accident_desc}

        **Final Decision:**
        - Provide a clear final decision of **who is responsible** for the accident based on the case details and general traffic law principles. Be sure to justify your decision based on either the details provided in the case or general traffic law.
        - Do not apologize or mention uncertainty, just provide a **confident** final decision.

    """
    
    # Format the prompt dynamically with the case information
    formatted_prompt = prompt_template.format(
        driver1_name=driver1_name,
        driver1_car_plate=driver1_car_plate,
        driver1_accident_desc=driver1_accident_desc,
        driver2_name=driver2_name,
        driver2_car_plate=driver2_car_plate,
        driver2_accident_desc=driver2_accident_desc
    )
    
    # Run the chain with the formatted prompt
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    try:
        response = chain.run(formatted_prompt)  # Running the chain with the formatted prompt
        return response
    except Exception as e:
        return f"Error during RAG chain execution: {str(e)}"



    # Initialize the prompt template using the input variables
    prompt = PromptTemplate(
        input_variables=["driver1_name", "driver1_car_plate", "driver1_accident_desc", "driver2_name", "driver2_car_plate", "driver2_accident_desc", "document_text"],
        template=prompt_template
    )

    # Format the prompt dynamically with the case information
    formatted_prompt = prompt.format(
        driver1_name=driver1_name,
        driver1_car_plate=driver1_car_plate,
        driver1_accident_desc=driver1_accident_desc,
        driver2_name=driver2_name,
        driver2_car_plate=driver2_car_plate,
        driver2_accident_desc=driver2_accident_desc,
        document_text=document_text
    )
    
    # Run the chain with the formatted prompt
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    try:
        response = chain.run(formatted_prompt)  # Running the chain with the formatted prompt
        return response
    except Exception as e:
        return f"Error during RAG chain execution: {str(e)}"


#-------------------------------------------------------------------



# Streamlit UI setup
def main():
    # Display the logo at the top
    display_logo()

    # Sidebar Section - providing instructions
    st.sidebar.title("Welcome to the Case Evaluator!")
    st.sidebar.markdown("Please fill out the forms below to evaluate the case.")
    st.sidebar.markdown("**Steps to follow:**")
    st.sidebar.markdown("1. Enter information about the drivers involved.")
    st.sidebar.markdown("2. Click on 'Evaluate Case' to get the decision.")
    st.sidebar.markdown("3. Review the case decision and indicate whether you agree.")
    st.sidebar.markdown("If you encounter any issues, feel free to contact support.")

    # Initialize session_state variables if not already set
    
    # if 'driver1_agree' not in st.session_state:
    #     st.session_state.driver1_agree = False
    # if 'driver2_agree' not in st.session_state:
    #     st.session_state.driver2_agree = False
    if 'case_evaluated' not in st.session_state:
        st.session_state.case_evaluated = False

    # Driver Information Section (Closed by default)
    st.markdown("<h3><strong>Driver 1 Information</strong></h3>", unsafe_allow_html=True)
    with st.expander("Enter Driver 1 Details", expanded=False):  # `expanded=False` makes it closed by default
        driver1_name = st.text_input("Driver 1 Name:")
        driver1_car_plate = st.text_input("Driver 1 Car Plate:")
        driver1_accident_desc = st.text_area("Driver 1 Accident Description", height=200)
    
    st.markdown("<h3><strong>Driver 2 Information</strong></h3>", unsafe_allow_html=True)
    with st.expander("Enter Driver 2 Details", expanded=False):  # `expanded=False` makes it closed by default
        driver2_name = st.text_input("Driver 2 Name:")
        driver2_car_plate = st.text_input("Driver 2 Car Plate:")
        driver2_accident_desc = st.text_area("Driver 2 Accident Description", height=200)

    # Evaluate Case Button
    if st.button("Evaluate Case") and not st.session_state.case_evaluated:
        if driver1_name and driver1_car_plate and driver1_accident_desc and driver2_name and driver2_car_plate and driver2_accident_desc:
            # Extract text from doc2.pdf
            pdf_text = extract_text_from_pdf("Doc2.pdf")
            chunks = split_text_into_chunks(pdf_text)
        
            # Process the case with the RAG chain
            vector_store = create_vector_store(chunks)
        
            try:
                # Get the response from the RAG chain
                response = create_rag_chain_with_prompt(
                    vector_store, 
                    driver1_name, 
                    driver1_car_plate, 
                    driver1_accident_desc, 
                    driver2_name, 
                    driver2_car_plate, 
                    driver2_accident_desc
                )

                # Display decision result in an expander
                with st.expander("Case Decision Result", expanded=True):
                    st.markdown("<h3><strong>Decision</strong></h3>", unsafe_allow_html=True)
                    st.write(response)
            
                # Ask Driver 1 for their agreement
                st.session_state.case_evaluated = True  # Mark the case as evaluated
        
            except Exception as e:
                st.error(f"Error in RAG decision: {str(e)}")
        else:
            st.error("Please fill in all the details.")


    # Driver 2 Agreement
    if st.session_state.case_evaluated:
        radio1 = st.radio(f"Driver 1, do you agree with the decision?", ['Yes', 'No'], key="driver1_agree", index=None)
        if 'driver1_agree' not in st.session_state:
            st.session_state.driver1_agree = radio1
        radio2 = st.radio(f"Driver 2, do you agree with the decision?", ['Yes', 'No'], key="driver2_agree", index=None)
        if 'driver2_agree' not in st.session_state:
            st.session_state.driver2_agree = radio2

        # Outcome based on driver agreements
        if st.session_state.driver1_agree == 'Yes' and st.session_state.driver2_agree == 'Yes': st.success("Both drivers agree with the decision. The case is closed.") 
        elif st.session_state.driver1_agree == 'No' or st.session_state.driver2_agree == 'No': st.warning("There is a contradiction. Please call the police.") 
        else: st.info("Waiting for both drivers to agree.")
        # if st.session_state.driver1_agree and st.session_state.driver2_agree:
        #     st.success("Both drivers agree with the decision. The case is closed.")
        # elif st.session_state.driver1_agree == 'No' or st.session_state.driver2_agree == 'No':
        #     st.warning("There is a contradiction. Please call the police.")

if __name__ == "__main__":
    main()
