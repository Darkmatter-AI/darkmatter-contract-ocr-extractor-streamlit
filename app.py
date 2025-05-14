import os
import re
import glob
import time
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral
from openai import OpenAI, OpenAIError
import pandas as pd
import tiktoken
from pydantic import BaseModel
import PyPDF2

st.set_page_config(
    page_title="SitusAMC AI Extractor",
    page_icon="situsamc_logo.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Ensure token_usage and step_times are initialized before any function uses them
if "token_usage" not in st.session_state:
    st.session_state["token_usage"] = {
        "Mistral OCR": {"request": 0, "answer": 0},
        "OpenAI Embeddings": {"request": 0, "answer": 0},
        "Field Suggestion": {"request": 0, "answer": 0},
        "Field Extraction": {"request": 0, "answer": 0},
    }
if "step_times" not in st.session_state:
    st.session_state["step_times"] = {
        "Mistral OCR": 0.0,
        "OpenAI Embeddings": 0.0,
        "Field Suggestion": 0.0,
        "Field Extraction": 0.0,
    }


def get_openai_client():
    """
    Get OpenAI client.
    Returns an OpenAI client.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key)


def get_mistral_client():
    """
    Get Mistral client.
    Returns a Mistral client.
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set.")
    return Mistral(api_key=api_key)


openai_client = get_openai_client()
mistral_client = get_mistral_client()


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


def get_embeddings(texts, max_tokens_per_batch=8000):
    """
    Use OpenAI's text-embedding-3-large model to get embeddings for a list of texts.
    Batches requests to stay under the max token limit. Each input should be a paragraph or similar chunk.
    Skips any chunk that exceeds the model's context window (8192 tokens).
    """
    start = time.perf_counter()
    max_context_tokens = 8192
    skipped_chunks = []

    # Helper to batch texts by token count
    def batch_texts(texts, max_tokens):
        batch = []
        batch_tokens = 0
        for i, t in enumerate(texts):
            t_tokens = count_tokens(str(t))
            if t_tokens > max_context_tokens:
                skipped_chunks.append((i, t))
                continue  # skip this chunk
            # If adding this paragraph would exceed the batch limit, yield the current batch
            if batch and batch_tokens + t_tokens > max_tokens:
                yield batch
                batch = []
                batch_tokens = 0
            batch.append(t)
            batch_tokens += t_tokens
        if batch:
            yield batch

    all_embeddings = []
    for batch in batch_texts(texts, max_tokens_per_batch):
        total_embed_tokens = sum([count_tokens(str(t)) for t in batch])
        st.session_state["token_usage"]["OpenAI Embeddings"][
            "request"
        ] += total_embed_tokens
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=batch,
        )
        all_embeddings.extend([item.embedding for item in embedding_response.data])
    elapsed = time.perf_counter() - start
    st.session_state["step_times"]["OpenAI Embeddings"] += elapsed

    # Warn if any chunks were skipped
    if skipped_chunks:
        st.warning(
            f"{len(skipped_chunks)} chunk(s) were skipped for embedding because they exceeded the 8192 token limit. Consider splitting large paragraphs."
        )
    return np.array(all_embeddings)


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


def markdown_heading_chunker(input_text):
    """
    Chunk markdown by headings. Consecutive headings are merged until a heading is followed by content (text, table, list, etc).
    Tables, bullet lists, and numbered lists are always chunked together as a single chunk.
    """

    lines = input_text.splitlines()
    chunks = []
    current_heading = []
    current_content = []
    in_table = False
    in_list = False
    table_lines = []
    list_lines = []

    def flush_content():
        nonlocal current_heading, current_content
        if current_content:
            chunk = "\n".join(current_heading + current_content)
            if chunk.strip():
                chunks.append(chunk.strip())
            current_heading = []
            current_content = []

    def flush_table():
        nonlocal table_lines
        if table_lines:
            chunks.append("\n".join(table_lines))
            table_lines = []

    def flush_list():
        nonlocal list_lines
        if list_lines:
            chunks.append("\n".join(list_lines))
            list_lines = []

    for ln in lines:
        stripped = ln.strip()
        # Heading
        if re.match(r"^#{1,6} ", stripped):
            # If we have content, flush it as a chunk
            flush_content()
            flush_table()
            flush_list()
            # If previous heading(s) had no content, merge this heading
            current_heading.append(stripped)
            continue
        # Table row (contains at least one | and not a list)
        if "|" in ln and not re.match(r"^\s*([-*]|\d+\.)", stripped):
            flush_content()
            flush_list()
            table_lines.append(ln)
            in_table = True
            continue
        else:
            if in_table:
                flush_table()
                in_table = False
        # List item (bullet or numbered)
        if re.match(r"^\s*([-*]|\d+\.)", stripped):
            flush_content()
            flush_table()
            list_lines.append(ln)
            in_list = True
            continue
        else:
            if in_list:
                flush_list()
                in_list = False
        # Normal content
        if stripped:
            # If we have merged headings and now see content, start content
            if current_heading:
                current_content.append(stripped)
            else:
                # No heading, just content
                current_content.append(stripped)
        else:
            # Blank line: treat as paragraph break
            if current_content:
                flush_content()
            if in_table:
                flush_table()
                in_table = False
            if in_list:
                flush_list()
                in_list = False
    # Flush any remaining content
    flush_content()
    flush_table()
    flush_list()
    return chunks


def load_field_templates_from_dir(directory="fields"):
    """
    Load field templates from a directory.
    directory: the directory to load field templates from
    Returns a dictionary of field templates.
    """
    templates = {}
    for file_path in glob.glob(os.path.join(directory, "*.txt")):
        title = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, "r", encoding="utf-8") as f:
            body = f.read().strip()
        templates[title] = body
    return templates


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


def run_mistral_ocr_with_progress(pdf_file, ocr_progress_bar):
    """
    Run Mistral OCR with progress bar.
    pdf_file: the PDF file to run OCR on
    ocr_progress_bar: the progress bar to update
    Returns a tuple of the OCR text and an error message if there is an error.
    """
    start = time.perf_counter()
    ocr_progress_bar.progress(10, text="Uploading PDF to Mistral...")
    uploaded_pdf = mistral_client.files.upload(
        file={
            "file_name": pdf_file.name,
            "content": pdf_file.getvalue(),
        },
        purpose="ocr",
    )
    ocr_progress_bar.progress(30, text="Getting signed URL...")
    signed_url = mistral_client.files.get_signed_url(file_id=uploaded_pdf.id)
    ocr_progress_bar.progress(50, text="Running OCR...")
    ocr_response = mistral_client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        },
    )
    ocr_progress_bar.progress(70, text="Extracting text from OCR response...")
    ocr_text = "\n\n".join([page.markdown for page in ocr_response.pages])
    elapsed = time.perf_counter() - start
    st.session_state["step_times"]["Mistral OCR"] += elapsed
    return ocr_text, None


def run_field_extraction():
    """
    Run field extraction with OpenAI (RAG)
    """
    extract_btn = st.button("Extract Fields with OpenAI (RAG)")
    extract_progress_placeholder = st.empty()
    if extract_btn and st.session_state["fields_input"].strip():
        extraction_start = time.perf_counter()
        extract_progress_bar = extract_progress_placeholder.progress(
            0, text="Extracting fields with OpenAI (RAG)..."
        )
        time.sleep(0.1)
        if not openai_client:
            st.error("OPENAI_API_KEY environment variable not set.")
            extract_progress_bar.progress(100, text="Error: OPENAI_API_KEY not set.")
            extract_progress_placeholder.empty()
        else:
            # Parse fields: field_name: data_type, description
            extract_progress_bar.progress(10, text="Parsing fields...")
            time.sleep(0.1)
            fields = []
            for field_line in st.session_state["fields_input"].splitlines():
                field_line = field_line.strip()
                if not field_line or field_line.startswith("#"):
                    continue  # skip comments and blank lines
                m = re.match(r"([^:]+):\s*([^,]+),\s*(.*)", field_line)
                if m:
                    fields.append(
                        {
                            "name": m.group(1).strip(),
                            "type": m.group(2).strip(),
                            "desc": m.group(3).strip(),
                        }
                    )
                else:
                    continue  # skip lines that don't match the expected field format
            extract_progress_bar.progress(
                25, text="Loading cached document chunks and embeddings..."
            )
            time.sleep(0.1)
            para_ocr_chunks = st.session_state.get("ocr_chunks")
            para_ocr_embeds = st.session_state.get("ocr_chunk_embeds")
            if para_ocr_chunks is None or para_ocr_embeds is None:
                st.error(
                    "Document chunks or embeddings not found. Please re-upload the PDF."
                )
                extract_progress_bar.progress(
                    100, text="Error: Missing cached embeddings."
                )
                extract_progress_placeholder.empty()
                return
            # Gradual field extraction: process each field one at a time
            field_extraction_rows = []
            table_placeholder = st.empty()
            num_fields = len(fields)
            TOP_K = 10  # Number of top chunks to use for each field's context
            for field_idx, field in enumerate(fields):
                extract_progress_bar.progress(
                    25 + int(50 * (field_idx + 1) / num_fields),
                    text=f"Finding relevant chunks and extracting for field {field_idx + 1}/{num_fields}...",
                )
                time.sleep(0.1)
                # Embed field prompt
                field_query = f"{field['name']} {field['desc']}"
                field_embed = get_embeddings([field_query])[0]
                # Cosine similarity
                sims = cosine_sim(np.array([field_embed]), para_ocr_embeds)[0]
                top_idx = np.argsort(sims)[-TOP_K:][::-1]  # top K chunks
                context = "\n".join([para_ocr_chunks[i] for i in top_idx])
                context = context[:5000]  # Optionally increase context length if needed
                # Prepare prompt for this field only
                prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are an extraction engine. For the given field, output a single line in the format: field_name,value,context. "
                            "If the value is a list or dictionary, output it a comma separated list. "
                            "Do not include any commentary or extra text. Only output one line per field."
                            "Example: contract_date,2025-01-01,The date the contract was signed.\n"
                            "Example: party_name,John Doe,The name of the party.\n"
                            "Example: amount,100000,The amount of the contract.\n"
                            "Example: contract_type,lease,The type of contract.\n"
                            "Example: expiration_date,2025-01-01,The date the contract expires.\n"
                            "Example: contract_type,lease,lease,lease\n"
                            "Example: contract_type,purchase,purchase,purchase\n"
                            "Example: contract_type,service,service,service\n"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Extract the value and context for the following field from the contract text.\n\n"
                            f"Field: {field['name']} (type: {field['type']}, desc: {field['desc']})\nContext:\n{context}"
                        ),
                    },
                ]
                prompt_token_count = sum([count_tokens(m["content"]) for m in prompt])
                st.session_state["token_usage"]["Field Extraction"][
                    "request"
                ] += prompt_token_count
                try:
                    field_extraction_response = openai_client.responses.create(
                        model="gpt-4.1",
                        input=prompt,
                    )
                    field_extraction_output = (
                        field_extraction_response.output[0].content[0].text.strip()
                    )
                    field_parts = field_extraction_output.split(",", 2)
                    if len(field_parts) == 3:
                        value = field_parts[1]
                        # Do not parse or modify the value, just display as-is (except for truncation)
                        value_display = value
                        max_value_length = 200
                        if len(value_display) > max_value_length:
                            value_display = value_display[:max_value_length] + "..."
                        field_extraction_rows.append(
                            {
                                "Field": field_parts[0],
                                "Value": value_display,
                                "Source": field_parts[2],
                            }
                        )
                        st.session_state["token_usage"]["Field Extraction"][
                            "answer"
                        ] += count_tokens(field_extraction_output)
                    # Update the table after each field
                    table_placeholder.dataframe(pd.DataFrame(field_extraction_rows))
                except OpenAIError as e:
                    st.error(f"Error extracting field {field['name']}: {e}")
                    continue
            extract_progress_bar.progress(100, text="Done!")
            extract_progress_placeholder.empty()
        st.session_state["step_times"]["Field Extraction"] += (
            time.perf_counter() - extraction_start
        )


load_dotenv()

st.image("situsamc_logo.jpeg", width=50)
st.title("Situs Contract Extractor")
st.markdown(
    "This tool extracts fields from a contract PDF. It uses Mistral OCR to extract the text, then uses OpenAI to extract the fields."
)

# User uploads a PDF file
uploaded_file = st.file_uploader("Upload a contract PDF", type=["pdf"])

progress = st.empty()
progress_bar = None  # pylint: disable=invalid-name


if uploaded_file:
    # Use a unique key for the uploaded file (e.g., name + size)
    file_key = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"
    if (
        "ocr_result" not in st.session_state
        or st.session_state.get("ocr_file_key") != file_key
    ):
        progress_bar = progress.progress(
            0, text="Uploading and processing with Mistral OCR..."
        )
        text, ocr_error = run_mistral_ocr_with_progress(uploaded_file, progress_bar)
        if ocr_error:
            st.error(ocr_error)
            progress_bar.progress(100, text=f"Error: {ocr_error}")
            progress.empty()
        else:
            progress_bar.progress(100, text="OCR complete!")
            progress.empty()
            st.session_state["ocr_result"] = text
            st.session_state["ocr_file_key"] = file_key
            # --- Compute and cache document chunks and embeddings during OCR ---
            paragraphs = markdown_heading_chunker(text)
            st.session_state["ocr_chunks"] = paragraphs
            # Compute embeddings for all chunks
            st.session_state["ocr_chunk_embeds"] = get_embeddings(paragraphs)
    else:
        text = st.session_state["ocr_result"]
        # Show a static progress bar at 100% to indicate completion
        progress_bar = progress.progress(100, text="OCR complete!")
        progress.empty()
    st.subheader("Extracted Text:")
    st.text_area("OCR Output", text, height=400)
    ocr_token_count = count_tokens(text)

    st.markdown("---")
    st.subheader("Extract Fields from Contract")

    # --- RAG Helper Functions ---

    # --- Field Suggestion Model for Parsing Suggestions ---
    class FieldSuggestion(BaseModel):
        """
        Field suggestion model for parsing suggestions
        """

        name: str
        type: str
        description: str
        example: str

    # --- Wrapper for list of FieldSuggestion (required for OpenAI structured output) ---
    class FieldSuggestions(BaseModel):
        """
        Wrapper for list of FieldSuggestion (required for OpenAI structured output)
        """

        fields: list[FieldSuggestion]

    # --- Field Extraction Templates (one prompt per file in fields/) ---

    field_templates = load_field_templates_from_dir()

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
    
    **Please enter your fields below:**
    """
    )

    # --- Three-column layout for template selection or field suggestion ---
    col1, col_or, col2 = st.columns([4, 1, 4])
    with col1:
        selected_template = st.selectbox(
            "Choose a field extraction template:",
            ["(None)"] + list(field_templates.keys()),
            index=0,
        )
        if selected_template != "(None)":
            st.session_state["fields_input"] = field_templates[selected_template]
    with col_or:
        st.markdown(
            "<div style='text-align:center; font-size: 1rem;'>or</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        Generate field extraction text from document
        """,
            unsafe_allow_html=True,
        )
        suggest_btn = st.button(
            "Generate",
            key="suggest_fields_btn",
        )
        suggest_progress = st.empty()
        if "fields_input" not in st.session_state:
            st.session_state["fields_input"] = ""
        if suggest_btn:
            suggest_start = time.perf_counter()
            suggest_progress.progress(0, text="Analyzing document to suggest fields...")
            # Use OpenAI to suggest fields
            extract_openai_key = os.environ.get("OPENAI_API_KEY")
            if not extract_openai_key:
                st.error("OPENAI_API_KEY environment variable not set.")
                suggest_progress.progress(100, text="Error: OPENAI_API_KEY not set.")
                suggest_progress.empty()
            else:
                extract_field_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are a contract field suggestion engine. For each field, output a single line in the format: field_name,type,description,example. "
                            "Do not include any commentary or extra text. Only output one line per field."
                            "Example: contract_date,date,The date the contract was signed,2025-01-01\n"
                            "Example: party_name,string,Name of the main party,John Doe\n"
                            "Example: amount,float,Total contract value,100000\n"
                            "Example: contract_type,string,Type of contract (e.g., lease, purchase, service),lease\n"
                            "Example: expiration_date,date,The date the contract expires,2025-01-01\n"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Given the following contract text, suggest a list of key fields that should be extracted. For each field, provide the field name, expected data type, a short description, and an example value.\n\nContract text:\n{text}",
                    },
                ]
                st.session_state["token_usage"]["Field Suggestion"][
                    "request"
                ] += count_tokens(str(extract_field_prompt))
                try:
                    suggest_progress.progress(
                        50, text="Contacting OpenAI for suggestions..."
                    )
                    response = openai_client.responses.create(
                        model="gpt-4.1",
                        input=extract_field_prompt,
                    )
                    suggest_progress.progress(80, text="Processing suggestions...")
                    output = response.output[0].content[0].text.strip()
                    rows = []
                    for line in output.splitlines():
                        parts = line.split(",", 3)
                        if len(parts) == 4:
                            rows.append(
                                {
                                    "Field Name": parts[0],
                                    "Type": parts[1],
                                    "Description": parts[2],
                                    "Example": parts[3],
                                }
                            )
                    FORMATTED = "\n".join(
                        [
                            f"{r['Field Name']}: {r['Type']}, {r['Description']}, {r['Example']}"
                            for r in rows
                        ]
                    )
                    st.session_state["fields_input"] = FORMATTED
                    st.session_state["token_usage"]["Field Suggestion"][
                        "answer"
                    ] += count_tokens(output)
                    suggest_progress.progress(100, text="Done!")
                    suggest_progress.empty()
                except OpenAIError as e:
                    st.error(f"Error suggesting fields: {e}")
                    st.session_state["fields_input"] = ""
                    suggest_progress.progress(
                        100, text="Error occurred during suggestion."
                    )
                    suggest_progress.empty()
            st.session_state["step_times"]["Field Suggestion"] += (
                time.perf_counter() - suggest_start
            )

    # Always show the text area for editing fields_input
    fields_input = st.text_area(
        "Enter fields to extract (one per line):",
        height=140,
        key="fields_input",
    )

    run_field_extraction()

    # --- Token Usage Tracking ---
    if "token_usage" not in st.session_state:
        st.session_state["token_usage"] = {
            "Mistral OCR": {"request": 0, "answer": 0},
            "OpenAI Embeddings": {"request": 0, "answer": 0},
            "Field Suggestion": {"request": 0, "answer": 0},
            "Field Extraction": {"request": 0, "answer": 0},
        }

    # --- Mistral OCR ---
    # For Mistral OCR, only the answer (extracted text) is relevant
    st.session_state["token_usage"]["Mistral OCR"]["answer"] += count_tokens(text)

    # --- Token Usage Table ---
    # Always show the token usage table at the end
    token_rows = [
        {
            "Step": step,
            "Request Tokens": usage["request"],
            "Answer Tokens": usage["answer"],
            "Time (s)": round(st.session_state["step_times"].get(step, 0.0), 2),
        }
        for step, usage in st.session_state["token_usage"].items()
    ]
    df_token = pd.DataFrame(token_rows)
    # Add a total row below the table
    total_request = df_token["Request Tokens"].sum()
    total_answer = df_token["Answer Tokens"].sum()

    # --- Estimated Cost Calculation ---
    # Pricing (2025):
    # Mistral OCR: $1 per 1,000 pages
    # OpenAI GPT-4.1: $10 per 1M input tokens, $30 per 1M output tokens
    # OpenAI Embeddings: $0.13 per 1M tokens (input only)
    # (You can adjust these as needed)

    # Try to get page count from uploaded_file if available
    page_count = None  # pylint: disable=invalid-name
    if "ocr_file_key" in st.session_state and uploaded_file:
        try:
            uploaded_file.seek(0)
            reader = PyPDF2.PdfReader(uploaded_file)
            page_count = len(reader.pages)
        except (PyPDF2.errors.PdfReadError, OSError):
            page_count = None  # pylint: disable=invalid-name

    # --- Cost Calculation for Each Step ---
    costs = []
    for idx, row in df_token.iterrows():
        step = row["Step"]
        req = row["Request Tokens"]
        ans = row["Answer Tokens"]
        if step == "Mistral OCR":
            # Only count if page_count is available
            cost = (page_count / 1000) if page_count else 0
        elif step == "OpenAI Embeddings":
            cost = req / 1_000_000 * 0.13
        elif step == "Field Suggestion":
            cost = req / 1_000_000 * 10 + ans / 1_000_000 * 30
        elif step == "Field Extraction":
            cost = req / 1_000_000 * 10 + ans / 1_000_000 * 30
        else:
            cost = 0
        costs.append(cost)

    df_token["Cost (USD)"] = [f"${c:.6f}" for c in costs]

    st.markdown("---")
    st.subheader("Token Usage Table (Request/Answer/Cost)")
    st.dataframe(df_token)
    st.markdown(
        f"**Total Request Tokens:** {total_request}  |  **Total Answer Tokens:** {total_answer}"
    )

    # Calculate costs
    mistral_ocr_cost = (  # pylint: disable=invalid-name
        page_count / 1000 if page_count else None
    )
    openai_input_cost = (
        (
            st.session_state["token_usage"]
            .get("Field Extraction", {})
            .get("request", 0)
            + st.session_state["token_usage"]
            .get("Field Suggestion", {})
            .get("request", 0)
            + st.session_state["token_usage"]
            .get("OpenAI Embeddings", {})
            .get("request", 0)
        )
        / 1_000_000
        * 10
    )
    openai_output_cost = (
        (
            st.session_state["token_usage"].get("Field Extraction", {}).get("answer", 0)
            + st.session_state["token_usage"]
            .get("Field Suggestion", {})
            .get("answer", 0)
        )
        / 1_000_000
        * 30
    )
    openai_embed_cost = (
        st.session_state["token_usage"].get("OpenAI Embeddings", {}).get("request", 0)
        / 1_000_000
        * 0.13
    )
    total_cost = 0  # pylint: disable=invalid-name
    if mistral_ocr_cost is not None:
        total_cost += mistral_ocr_cost
    if openai_input_cost is not None:
        total_cost += openai_input_cost
    if openai_output_cost is not None:
        total_cost += openai_output_cost
    if openai_embed_cost is not None:
        total_cost += openai_embed_cost

    st.markdown(
        f"**Total Request Tokens:** {total_request}  |  **Total Answer Tokens:** {total_answer}  |  **Total Estimated Cost:** ~${total_cost:.2f}"
    )

# Footer
st.markdown(
    """
    <hr style='margin-top:2em;margin-bottom:0.5em;'>
    <div style='text-align:center; color: #888; font-size: 0.9em;'>
        made by <a href='https://darkmatter.is' target='_blank' style='color:#888;text-decoration:none;'>Darkmatter© darkmatter.is</a>
    </div>
    """,
    unsafe_allow_html=True,
)
