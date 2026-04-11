import streamlit as st
import pandas as pd
import requests
import json

from state.session import setup_session_state, is_chat_ready
from components.chat import (
  render_chat_history,
  render_download_chat_history,
  render_uploaded_files_expander,
  render_user_input
)
from components.sidebar import (
  render_model_selector,
  render_view_selector,
  sidebar_file_upload,
  sidebar_provider_change_check,
  sidebar_utilities
)
from components.inspector import render_inspect_query


def main():
  st.set_page_config(page_title="EDR | Characterization Dashboard", page_icon="⚡", layout="wide")
  #st.title("👽 RAG PDFBot")
  #st.caption("Chat with multiple PDFs :books:")

  # --- CUSTOM CSS ---
  st.markdown(
    """
    <style>
        /* Target the Streamlit sidebar and force a new width */
        [data-testid="stSidebar"] {
            min-width: 600px;
            max-width: 600px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

  setup_session_state()

  if "messages" not in st.session_state:
    st.session_state.messages = []
  if "extracted_data" not in st.session_state:
    # A placeholder dataframe for the right-hand dashboard
    st.session_state.extracted_data = pd.DataFrame(columns=["Parameter", "Value", "Condition"])

  if st.session_state.get("chat_history"):
    render_download_chat_history()

  with st.sidebar:
    st.title("⚡ EDR")
    st.markdown("Automated Datasheet Extraction")

    # Workspace Selection (For Vector Filtering later)
    st.subheader("1. Select Workspace")
    workspace = st.selectbox(
      "Active Component Domain:",
      ["Op-Amps & Amplifiers", "Photodetectors & Lasers", "RF & Wireless", "Power Management"]
    )

    st.subheader("2. Upload Datasheet")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
      if st.button("Process Document"):
        with st.spinner("Running Layout Analysis & Vectorization..."):
          #TODO: Replace with your actual FastAPI upload endpoint request
          # files = {"file": uploaded_file.getvalue()}
          # res = requests.post("http://localhost:8000/upload", files=files)
          st.success("Datasheet processed and added to workspace!")

    st.subheader("3. Session Controls")
    if st.button("🧹 Clear Workspace Cache"):
      st.session_state.messages = []
      st.session_state.extracted_data = pd.DataFrame(columns=["Parameter", "Value", "Condition"])
      st.rerun()
    
    # --- MAIN SPLIT-SCREEN LAYOUT ---
    # Column 1 takes 50% of the screen, Column 2 takes 50%
    chat_col, dashboard_col = st.columns([1, 1], gap="large")

    # --- LEFT COLUMN: THE CHAT INTERFACE ---
    with chat_col:
      st.header(f"Query Engine: {workspace}")

      # Display chat history
      for message in st.session_state.messages:
        with st.chat_message(message["role"]):
          st.markdown(message["content"])

      # Chat Input
      if prompt := st.chat_input("Ask for absolute maximum ratings, gain bandwidth, etc..."):
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Querying Vector Database..."):
                # TODO: Replace with your actual FastAPI chat endpoint request
                # payload = {"query": prompt, "workspace": workspace}
                # response = requests.post("http://localhost:8000/chat", json=payload).json()
                # answer = response["answer"]
                
                # Mock response for UI testing
                answer = "Based on the structural analysis of the characteristics table, the Common-Mode Input Voltage Range is $V_{SS} - 0.3V$ to $V_{DD} - 1.2V$."
                st.markdown(answer)
                
                # Update UI State
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # --- MOCK DATA EXTRACTION TRIGGER ---
                # In production, if the user asks for parameters, your FastAPI backend 
                # should return structured JSON alongside the text answer. 
                if "range" in prompt.lower() or "parameter" in prompt.lower():
                    st.session_state.extracted_data = pd.DataFrame({
                        "Parameter": ["V_CMR (Min)", "V_CMR (Max)"],
                        "Value": ["V_SS - 0.3V", "V_DD - 1.2V"],
                        "Condition": ["V_DD = 5V", "V_DD = 5V"]
                    })
                    st.rerun()

  # --- RIGHT COLUMN: THE ENGINEERING DASHBOARD ---
  with dashboard_col:
    st.header("Extracted Parameters")
    
    # The Live Data Table
    st.markdown("Detected electrical characteristics from current query:")
    st.dataframe(
        st.session_state.extracted_data, 
        width='stretch',
        hide_index=True
    )
    
    st.divider()
    
    # EDA Export Integration
    st.subheader("EDA Integration")
    st.markdown("Export parameters for circuit simulation:")
    
    # Generate a mock .model directive based on the dataframe
    spice_model_text = "* Auto-generated SPICE Directive from ElecDocReader\n.model AutoOpAmp auto (\n"
    for index, row in st.session_state.extracted_data.iterrows():
        spice_model_text += f"+ {row['Parameter'].replace(' ', '_')}={row['Value']} ; Condition: {row['Condition']}\n"
    spice_model_text += ")"
    
    # The 1-Click Download Button
    st.download_button(
        label="📥 Download .model for LTSpice",
        data=spice_model_text,
        file_name="component_params.model",
        mime="text/plain",
        type="primary" # Uses the accent color from config.toml
    )
    
    # Additional Context Box
    with st.expander("View Raw Layout Source"):
        st.info("When UnstructuredPDFLoader is fully integrated, the isolated bounding box image of the table or graph used to generate this data will render here.")

  #   with st.expander("⚙️ Configuration", expanded=True):
  #     model_provider, model = render_model_selector()
  #     sidebar_file_upload(model_provider)
  #     sidebar_provider_change_check(model_provider, model)

  #   view_option = render_view_selector()
  #   sidebar_utilities()

  # if not st.session_state.get(f"uploaded_files_{st.session_state.uploader_key}", []):
  #   st.info("📄 Please upload and submit PDFs to start chatting.")

  # if st.session_state.get("unsubmitted_files", False):
  #   st.warning("📄 New PDFs uploaded. Please submit before chatting.")

  # if st.session_state.get("chat_ready") and st.session_state.get("pdf_files", []):
  #   render_uploaded_files_expander()

  # if view_option == "💬 Chat":
  #   if st.session_state.get("chat_history", []):
  #     render_chat_history()

  #   if is_chat_ready():
  #     render_user_input(model_provider, model)
  # elif view_option == "🔬 Inspector":
  #   if is_chat_ready():
  #     render_inspect_query(model_provider)

if __name__ == "__main__":
    main()
