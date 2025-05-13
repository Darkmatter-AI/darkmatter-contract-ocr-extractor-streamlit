import os
import re
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral
from openai import OpenAI, OpenAIError
import pandas as pd
import tiktoken
import tokencost


def count_tokens(token_text):
    """
    Count the number of tokens in a text using tiktoken.
    token_text: the text to count tokens for
    Returns the number of tokens in the text.
    """
    try:
        enc = tiktoken.encoding_for_model("gpt-4.1")
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(token_text))


load_dotenv()

st.title("Contract OCR Extractor (Mistral)")

# User uploads a PDF file
uploaded_file = st.file_uploader("Upload a contract PDF", type=["pdf"])

progress = st.empty()
progress_bar = None  # pylint: disable=invalid-name

if uploaded_file:
    progress_bar = progress.progress(
        0, text="Uploading and processing with Mistral OCR..."
    )
    try:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            st.error("MISTRAL_API_KEY environment variable not set.")
            progress_bar.progress(100, text="Error: MISTRAL_API_KEY not set.")
            progress.empty()
        else:
            client = Mistral(api_key=api_key)
            # Upload the PDF to Mistral
            progress_bar.progress(10, text="Uploading PDF to Mistral...")
            uploaded_pdf = client.files.upload(
                file={
                    "file_name": uploaded_file.name,
                    "content": uploaded_file.getvalue(),  # getvalue() returns bytes
                },
                purpose="ocr",
            )
            progress_bar.progress(30, text="Getting signed URL...")
            # Get signed URL
            signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
            progress_bar.progress(50, text="Running OCR...")
            # Run OCR
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                },
            )
            progress_bar.progress(70, text="Extracting text from OCR response...")
            # Display extracted text (markdown from all pages)
            text = "\n\n".join([page.markdown for page in ocr_response.pages])
            st.subheader("Extracted Text:")
            st.text_area("OCR Output", text, height=400)
            # Show token count for OCR output
            ocr_token_count = count_tokens(text)
            st.info(f"OCR extracted text token count: {ocr_token_count}")

            st.markdown("---")
            st.subheader("Extract Fields from Contract")

            # --- RAG Helper Functions ---

            def split_into_paragraphs(input_text):
                """
                Splits the text into paragraphs.
                Paragraphs are separated by double newlines or single newlines if paragraph is long.
                Paragraphs are stripped of whitespace.
                Paragraphs are returned as a list.
                """
                # Split by double newlines or single newlines if paragraph is long
                paras = re.split(r"\n\s*\n", input_text)
                return [p.strip() for p in paras if p.strip()]

            def get_embeddings(texts, openai_api_key=None):
                """
                Use OpenAI's text-embedding-3-large model to get embeddings for a list of texts.
                openai_api_key: should be set in the environment as OPENAI_API_KEY
                """
                if openai_api_key is None:
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                if not openai_api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set.")
                embedding_client = OpenAI(api_key=openai_api_key)
                embedding_response = embedding_client.embeddings.create(
                    model="text-embedding-3-large",
                    input=texts,
                )
                # response.data is a list of objects with 'embedding' key
                return np.array([item.embedding for item in embedding_response.data])

            def cosine_sim(a, b):
                """
                Calculates the cosine similarity between two vectors.
                a and b are numpy arrays.
                Returns a numpy array of cosine similarities.
                """
                a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
                b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
                return np.dot(a, b.T)

            def chunk_sentences(input_text, sentences_per_chunk=5, overlap=2):
                """
                Splits the text into overlapping sentence chunks for better RAG performance.
                sentences_per_chunk: number of sentences per chunk
                overlap: number of sentences to overlap between chunks
                Returns a list of text chunks.
                """
                # Split text into sentences using regex
                sentences = re.split(r"(?<=[.!?]) +", input_text)
                chunks = []
                i = 0
                while i < len(sentences):
                    chunk = " ".join(sentences[i : i + sentences_per_chunk])
                    if chunk.strip():
                        chunks.append(chunk.strip())
                    i += sentences_per_chunk - overlap
                return chunks

            def hybrid_markdown_chunker(
                input_text, max_words=120, sentences_per_chunk=5, overlap=2
            ):
                """
                Hybrid chunking for markdown:
                - Split by double newlines to get blocks (paragraphs, tables, lists)
                - Keep tables (lines with |) and lists (lines starting with -, *, or 1.) as single chunks
                - For long paragraphs, split into sentence-based chunks
                """

                blocks = re.split(r"\n\s*\n", input_text)
                chunks = []
                for block in blocks:
                    block = block.strip()
                    if not block:
                        continue
                    lines = block.splitlines()
                    # Table: at least 2 lines with '|' and at least one line starts with '|'
                    if sum("|" in line for line in lines) >= 2 and any(
                        line.strip().startswith("|") for line in lines
                    ):
                        chunks.append(block)
                        continue
                    # List: most lines start with -, *, or numbered
                    if sum(
                        bool(re.match(r"^(\s*[-*]|\s*\d+\.)", line)) for line in lines
                    ) > 0.5 * len(lines):
                        chunks.append(block)
                        continue
                    # If block is long, split into sentence chunks
                    words = block.split()
                    if len(words) > max_words:
                        sentences = re.split(r"(?<=[.!?]) +", block)
                        i = 0
                        while i < len(sentences):
                            chunk = " ".join(sentences[i : i + sentences_per_chunk])
                            if chunk.strip():
                                chunks.append(chunk.strip())
                            i += sentences_per_chunk - overlap
                    else:
                        chunks.append(block)
                return chunks

            def markdown_heading_chunker(input_text, max_words=300):
                """
                Chunk markdown by headings. Each chunk is a section under a heading (e.g., #, ##, etc).
                If a section is very long, split by paragraphs but keep the heading at the top of each chunk.
                Tables and lists are preserved within their section.
                """

                # Split on headings, keeping the heading with the content
                sections = re.split(r"(?=^#{1,6} )", input_text, flags=re.MULTILINE)
                chunks = []
                for section in sections:
                    section = section.strip()
                    if not section:
                        continue
                    words = section.split()
                    if len(words) <= max_words:
                        chunks.append(section)
                    else:
                        # For long sections, split by double newlines (paragraphs)
                        chunked_paragraphs = re.split(r"\n\s*\n", section)
                        heading = ""
                        if chunked_paragraphs and chunked_paragraphs[0].startswith("#"):
                            heading = chunked_paragraphs[0]
                            chunked_paragraphs = chunked_paragraphs[1:]
                        current_chunk = [heading] if heading else []
                        current_len = len(heading.split())
                        for para in chunked_paragraphs:
                            para_words = para.split()
                            if (
                                current_len + len(para_words) > max_words
                                and current_chunk
                            ):
                                chunks.append(
                                    "\n\n".join([c for c in current_chunk if c])
                                )
                                current_chunk = [heading, para] if heading else [para]
                                current_len = (
                                    len(heading.split()) + len(para_words)
                                    if heading
                                    else len(para_words)
                                )
                            else:
                                current_chunk.append(para)
                                current_len += len(para_words)
                        if current_chunk:
                            chunks.append("\n\n".join([c for c in current_chunk if c]))
                return chunks

            # Suggest fields button and logic
            suggest_btn = st.button("Suggest Fields from Document")
            suggest_progress = st.empty()
            if "fields_input" not in st.session_state:
                st.session_state["fields_input"] = ""

            if suggest_btn:
                progress_bar2 = suggest_progress.progress(
                    0, text="Analyzing document to suggest fields..."
                )
                # Use OpenAI to suggest fields
                openai_key = os.environ.get("OPENAI_API_KEY")
                if not openai_key:
                    st.error("OPENAI_API_KEY environment variable not set.")
                    progress_bar2.progress(100, text="Error: OPENAI_API_KEY not set.")
                    suggest_progress.empty()
                else:
                    client = OpenAI(api_key=openai_key)
                    PROMPT = (
                        "Given the following contract text, suggest a list of key fields that should be extracted. "
                        "For each field, provide the field name, expected data type, and a short description. "
                        "Return the result as a list, one field per line, in the format: field_name: data_type, description, example_value.\n"
                        "\nExample:\n"
                        "contract_date: date, The date the contract was signed, 2025-01-01\n"
                        "party_name: string, Name of the main party, John Doe\n"
                        "amount: float, Total contract value, 100000\n"
                        "contract_type: string, Type of contract (e.g., lease, purchase, service), lease\n"
                        "expiration_date: date, The date the contract expires, 2025-01-01\n"
                        f"\nContract text:\n{text}"
                    )
                    try:
                        progress_bar2.progress(
                            50, text="Contacting OpenAI for suggestions..."
                        )
                        response = client.responses.create(
                            model="gpt-4.1",
                            input=PROMPT,
                            max_output_tokens=256,
                        )
                        progress_bar2.progress(80, text="Processing suggestions...")
                        suggestion = response.output[0].content[0].text.strip()
                        # Keep the full suggestion (field: type, description)
                        st.session_state["fields_input"] = suggestion
                        progress_bar2.progress(100, text="Done!")
                        suggest_progress.empty()
                    except OpenAIError as e:
                        st.error(f"OpenAI API error (suggest fields): {e}")
                        progress_bar2.progress(
                            100, text="Error occurred during suggestion."
                        )
                        suggest_progress.empty()

            st.markdown(
                """
            **Field format:**
            
            Each field should be on its own line, in the following format:
            
            `field_name: data_type, description, example_value`
            
            **Example:**
            
            contract_date: date, The date the contract was signed, 2025-01-01
            party_name: string, Name of the main party, John Doe
            amount: float, Total contract value, 100000
            contract_type: string, Type of contract (e.g., lease, purchase, service), lease
            expiration_date: date, The date the contract expires, 2025-01-01
            
            Please enter your fields below:
            """
            )
            fields_input = st.text_area(
                "Enter fields to extract (one per line):",
                height=140,
                key="fields_input",
            )
            # Show token count for field request input
            field_input_token_count = count_tokens(fields_input)
            st.info(f"Field request input token count: {field_input_token_count}")

            extract_btn = st.button("Extract Fields with OpenAI (RAG)")
            extract_progress_placeholder = st.empty()

            if extract_btn and fields_input.strip():
                progress_bar = extract_progress_placeholder.progress(
                    0, text="Extracting fields with OpenAI (RAG)..."
                )
                openai_key = os.environ.get("OPENAI_API_KEY")
                if not openai_key:
                    st.error("OPENAI_API_KEY environment variable not set.")
                    progress_bar.progress(100, text="Error: OPENAI_API_KEY not set.")
                    progress.empty()
                else:
                    # Parse fields: field_name: data_type, description
                    fields = []
                    for line in fields_input.splitlines():
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue  # skip comments and blank lines
                        m = re.match(r"([^:]+):\s*([^,]+),\s*(.*)", line)
                        if m:
                            fields.append(
                                {
                                    "name": m.group(1).strip(),
                                    "type": m.group(2).strip(),
                                    "desc": m.group(3).strip(),
                                }
                            )
                        else:
                            fields.append({"name": line, "type": "string", "desc": ""})

                    # 1. Chunk contract text
                    paragraphs = markdown_heading_chunker(text)
                    # 2. Embed all paragraphs
                    progress_bar.progress(10, text="Embedding document chunks...")
                    para_embeds = get_embeddings(paragraphs)

                    # 3. For each field, find top relevant chunks and extract
                    results = {}
                    client = OpenAI(api_key=openai_key)
                    table_rows = []
                    table_placeholder = st.empty()
                    prompt_token_counts = []
                    for idx, field in enumerate(fields):
                        progress_bar.progress(
                            10 + int(80 * idx / max(1, len(fields))),
                            text=f"Processing field: {field['name']}",
                        )
                        # Embed field prompt
                        field_query = f"{field['name']} {field['desc']}"
                        field_embed = get_embeddings([field_query])[0]
                        # Cosine similarity
                        sims = cosine_sim(np.array([field_embed]), para_embeds)[0]
                        top_idx = np.argsort(sims)[-3:][::-1]  # top 3 chunks
                        context = "\n".join([paragraphs[i] for i in top_idx])
                        # Prompt for extraction
                        prompt = (
                            f"Extract the value for the field '{field['name']}' (type: {field['type']}, desc: {field['desc']}) "
                            f"from the following contract text. Return only the value, nothing else.\n\n{context}"
                        )
                        prompt_tokens = count_tokens(prompt)
                        prompt_token_counts.append((field["name"], prompt_tokens))
                        try:
                            field_response = client.responses.create(
                                model="gpt-4.1",
                                input=prompt,
                                max_output_tokens=64,
                            )
                            value = field_response.output[0].content[0].text.strip()
                        except OpenAIError as e:
                            value = f"Error: {e}"
                        results[field["name"]] = value
                        # Add a snippet of the context as the source
                        snippet = context[:300] + ("..." if len(context) > 300 else "")
                        table_rows.append(
                            {"Field": field["name"], "Value": value, "Source": snippet}
                        )
                        table_placeholder.dataframe(pd.DataFrame(table_rows))
                    # Finalize progress bar
                    table_placeholder.dataframe(pd.DataFrame(table_rows))
                    progress_bar.progress(100, text="Done!")
                    progress.empty()
                    # Show token counts for each field prompt
                    st.markdown("**Token count for each field extraction prompt:**")
                    for fname, tcount in prompt_token_counts:
                        st.info(f"{fname}: {tcount} tokens")

            # --- Cost Calculation Table ---
            def calculate_step_costs(
                embedding_texts,
                prompt_texts,
                completion_texts,
                embedding_model,
                prompt_model,
            ):
                embedding_cost = tokencost.calculate_prompt_cost(
                    embedding_texts, embedding_model
                )
                prompt_cost = tokencost.calculate_prompt_cost(
                    prompt_texts, prompt_model
                )
                completion_cost = tokencost.calculate_completion_cost(
                    completion_texts, prompt_model
                )
                return [
                    {
                        "Step": "Embedding",
                        "Model": embedding_model,
                        "Cost (USD)": embedding_cost,
                    },
                    {
                        "Step": "Prompt",
                        "Model": prompt_model,
                        "Cost (USD)": prompt_cost,
                    },
                    {
                        "Step": "Completion",
                        "Model": prompt_model,
                        "Cost (USD)": completion_cost,
                    },
                ]

            # After all processing, show the cost table
            if (
                "para_embeds" in locals()
                and "fields" in locals()
                and "results" in locals()
            ):
                # Example: calculate costs for the main steps
                embedding_model = "text-embedding-3-large"
                prompt_model = "gpt-4.1"
                embedding_texts = paragraphs  # list of text chunks embedded
                prompt_texts = [
                    f"Extract the value for the field '{field['name']}' (type: {field['type']}, desc: {field['desc']}) from the following contract text. Return only the value, nothing else.\n\n..."
                    for field in fields
                ]
                completion_texts = list(results.values())
                cost_rows = calculate_step_costs(
                    embedding_texts,
                    prompt_texts,
                    completion_texts,
                    embedding_model,
                    prompt_model,
                )
                st.markdown("---")
                st.subheader("Cost Breakdown Table (USD)")
                st.dataframe(pd.DataFrame(cost_rows))
    except OpenAIError as e:
        st.error(f"An error occurred: {e}")
        progress_bar.progress(100, text="Error occurred during processing.")
        progress.empty()
