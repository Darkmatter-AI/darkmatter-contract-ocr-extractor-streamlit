import streamlit as st
import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

st.title("Contract OCR Extractor (Mistral)")

# User uploads a PDF file
uploaded_file = st.file_uploader("Upload a contract PDF", type=["pdf"])

if uploaded_file:
    st.info("Uploading and processing with Mistral OCR...")
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        st.error("MISTRAL_API_KEY environment variable not set.")
    else:
        client = Mistral(api_key=api_key)
        # Upload the PDF to Mistral
        # Streamlit's uploaded_file is a file-like object, but Mistral may expect bytes or a file-like object
        uploaded_pdf = client.files.upload(
            file={
                "file_name": uploaded_file.name,
                "content": uploaded_file.getvalue(),  # getvalue() returns bytes
            },
            purpose="ocr"
        )
        # Get signed URL
        signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
        # Run OCR
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            }
        )
        # Display extracted text (markdown from all pages)
        text = "\n\n".join([page.markdown for page in ocr_response.pages])
        st.subheader("Extracted Text:")
        st.text_area("OCR Output", text, height=400)

        st.markdown("---")
        st.subheader("Extract Fields from Contract")
        fields_input = st.text_area("Enter fields to extract (one per line):", height=100)
        extract_btn = st.button("Extract Fields with OpenAI")

        if extract_btn and fields_input.strip():
            from openai import OpenAI
            from pydantic import create_model
            import json
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            if not os.environ.get("OPENAI_API_KEY"):
                st.error("OPENAI_API_KEY environment variable not set.")
            else:
                fields = [f.strip() for f in fields_input.splitlines() if f.strip()]
                # Dynamically create a Pydantic model for the fields (no default values)
                FieldModel = create_model(
                    "ContractFields",
                    **{field: (str, ...) for field in fields}  # use Ellipsis to indicate required, no default
                )
                prompt = (
                    "Extract the following fields from the contract text below. "
                    "Return a JSON object with the field names as keys and their values as found in the text. "
                    "If a field is not found, use an empty string. Respond only with a valid JSON object.\n"
                    f"Fields: {fields}\n"
                    f"Contract text:\n{text}"
                )
                try:
                    response = client.responses.parse(
                        model="gpt-4.1",
                        input=prompt,
                        text_format=FieldModel
                    )
                    data = response.output_parsed
                    if data:
                        st.subheader("Extracted Fields:")
                        st.table([{**{"Field": k, "Value": v}} for k, v in data.model_dump().items()])
                    # Handle refusal
                    if hasattr(response, "refusal") and response.refusal:
                        st.warning(f"Model refused to answer: {response.refusal}")
                except Exception as e:
                    st.error(f"OpenAI API error: {e}") 