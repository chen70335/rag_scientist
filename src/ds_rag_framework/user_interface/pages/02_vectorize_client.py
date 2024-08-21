import streamlit as st
import os
import tempfile

from ds_rag_framework.vectorizer.vectorizer import DocVectorizer
from ds_rag_framework.llms.llm_picker import LLMPicker
from ds_rag_framework.data_loader.load import HTMLDataLoader, PDFDataLoader


def save_uploaded_file(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


if __name__ == '__main__':
    st.title("HealthEdge Knowledge Database Vectorization Interface!")
    document_type = st.selectbox(
        label="Document Type",
        placeholder="Select document type...",
        options=["pdf", "html", "csv"],
        key="document_type"
    )
    index_storage_name = st.text_input("Index Storage Name", key="index_storage_name", placeholder="Ex: test_agent_pdf_index")
    storage_location = st.selectbox("Storage Location", ["local", "azure_cognitive_search", 'google_vertex_ai'], key="storage_location")
    if document_type == 'html':
        input_urls = st.text_area("Input URLs", key="input_urls", placeholder="Please provide urls separated by commas. Ex: https://www.healthedge.com/healthrules-payor-24.1, https://www.healthedge.com/healthrules-payor-24.2, https://www.healthedge.com/healthrules-payor-24.3")
    elif document_type == 'pdf':
        input_pdfs = st.file_uploader("Input PDFs", key="input_pdfs", type=["pdf"], accept_multiple_files=True)
    elif document_type == 'csv':
        input_csvs = st.file_uploader("Input CSVs", key="input_csvs", type=["csv"], accept_multiple_files=True)
    metadata_fields_num = st.number_input("Add Metadata Fields: ", key="metadata_fields_num", min_value=0, max_value=10)

    if metadata_fields_num != 0:
        metadata_fields_cols = st.columns(metadata_fields_num)
        for i in range(metadata_fields_num):
            with metadata_fields_cols[i]:
                metadata_field_key = st.text_input(f"Metadata Field Key {i+1}", key=f"metadata_field_key_{i+1}",
                                                   placeholder="Ex: version")
                metadata_field_value = st.text_input(f"Metadata Field Value {i+1}", key=f"metadata_field_value_{i+1}",
                                                     placeholder="Ex: 24.1")
    merge = st.checkbox("Merging into Existing Index?", key="merge")
    llm_model = st.selectbox("LLM Model", ["AzureOpenAI", "OpenAI", "Vertex", "HuggingFace_API"], key="llm_model")
    chunk_size = st.selectbox("Chunk Size", [256, 512, 1024, 2048], key="chunk_size")
    submitted = st.button("Submit")

    if submitted:
        with st.spinner('Creating Vector Database...'):
            # 1. load documents
            if document_type == 'html':
                html_loader = HTMLDataLoader(input_urls=input_urls)
                documents = html_loader.load_with_simple_text_extraction()
            elif document_type == 'pdf':
                print(f"here are the input files: {input_pdfs[0].readinto}")
                input_pdfs = [save_uploaded_file(file) for file in input_pdfs]
                print(f"here are the input files: {input_pdfs[0]}")
                pdf_loader = PDFDataLoader(input_files=input_pdfs)
                documents = pdf_loader.load_with_llama_parse()
            elif document_type == 'csv':
                raise NotImplementedError("CSV data loader is not implemented yet.")


            llm_picker = LLMPicker(llm_provider=llm_model.lower(), chunk_size=chunk_size)
            vectorizer = DocVectorizer(llm_picker=llm_picker)
            # access metadata fields here
            for i in range(metadata_fields_num):
                print(f"metadata field key: {st.session_state[f'metadata_field_key_{i+1}']}")

            if storage_location == 'local':
                st.session_state.vector_index = vectorizer.build_and_store_local(documents=documents, index_path=index_storage_name, merge=merge)
            elif storage_location == 'azure_cognitive_search':
                st.session_state.vector_index = vectorizer.build_and_store_azure(documents=documents, index_name=index_storage_name, merge=merge,
                                                    filterable_metadata_fields=['version'] if document_version else None)
        st.success("Successfully created vector database!")
        print(f"session state vector index: {st.session_state.vector_index}")