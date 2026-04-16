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
    # Dataframe for the right-hand dashboard to store values
    st.session_state.extracted_data = pd.DataFrame(columns=["Parameter", "Value", "Condition", "Page"])

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
    if st.button("Process Document"):
      with st.spinner("Running Layout Analysis & Vectorization..."):
        try:
          # Format the file for FastAPI's UploadFile
          files = [("files", (uploaded_file.name, uploaded_file.getvalue(), "application/pdf"))]
          # Hardcode the provider
          data = {"model_provider": "groq"} 
          
          # Hit the actual backend endpoint!
          res = requests.post("http://localhost:8000/upload_and_process_pdfs", files=files, data=data)
          
          if res.status_code == 200:
            st.success("Datasheet processed and added to workspace!")
          else:
            st.error(f"Failed to process: {res.text}")
        except Exception as e:
          st.error(f"Backend connection failed: {e}")

    st.subheader("3. Session Controls")
    if st.button("🧹 Clear Workspace Cache"):
      st.session_state.messages = []
      st.session_state.extracted_data = pd.DataFrame(columns=["Parameter", "Value", "Condition", "Page"])
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

    
    prompt = st.chat_input("Ask for absolute maximum ratings, gain bandwidth, etc...")
    # Chat Input
    if prompt:
      # Add user message to UI
      st.session_state.messages.append({"role": "user", "content": prompt})
      with st.chat_message("user"):
          st.markdown(prompt)

      # Generate Assistant Response
      with st.chat_message("assistant"):
        with st.spinner("Querying Vector Database..."):
          # TODO: Replace with your actual FastAPI chat endpoint request
          payload = {"query": prompt, "workspace": workspace}

          try:
            res = requests.post("http://localhost:8000/extract", json=payload)
            response_data = res.json()
              
            answer = response_data.get("answer", "Error generating response.")
            parameters = response_data.get("parameters", [])
                
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
                
            # If the LLM found parameters, update the right-hand dashboard!
            if parameters:
              st.session_state.extracted_data = pd.DataFrame(parameters)
              st.rerun() # Force the dashboard to refresh with the new data
                      
          except Exception as e:
            st.error(f"Backend connection failed: {e}")

  # --- RIGHT COLUMN: THE ENGINEERING DASHBOARD ---
  with dashboard_col:
    st.header("Extracted Parameters")
    
    # The Live Data Table
    st.markdown("Detected electrical characteristics from current query:")
    st.dataframe(
      st.session_state.extracted_data, 
      use_container_width=True,
      hide_index=True
    )
    
    st.divider()
    
    # EDA Export Integration
    st.subheader("EDA Integration")
    st.markdown("Export parameters for circuit simulation:")
    
    # Generate a mock .model directive based on the dataframe
    spice_model_text = "* Auto-generated SPICE Directive from ElecDocReader\n.model AutoOpAmp auto (\n"
    for index, row in st.session_state.extracted_data.iterrows():
      page_info = row.get('Page', 'N/A')
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
    
if __name__ == "__main__":
    main()
               