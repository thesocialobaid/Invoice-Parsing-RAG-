import streamlit as st
from dotenv import load_dotenv
import invoiceutil as iu
import json

def main():
    load_dotenv()

    st.set_page_config(page_title="Invoice Extraction Bot")
    st.title("🧾 Invoice Extraction Bot")
    st.subheader("I can help you extract structured invoice data from invoices using AI!")

    # Uploading the invoices (PDF only)
    pdf_files = st.file_uploader(
        "Upload invoices here (PDF only)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Extract Data"):
        if pdf_files:
            with st.spinner('🔄 Processing invoices...'):
                extracted_data = iu.create_docs(pdf_files)

                st.subheader("📄 Extracted Invoice Data:")
                for item in extracted_data:
                    for filename, content in item.items():
                        st.markdown(f"### 📂 {filename}")
                        try:
                            # Attempt to parse and display as JSON
                            parsed_content = json.loads(content)
                            st.json(parsed_content)
                        except json.JSONDecodeError as e:
                            st.error(f"JSON Parse Error: {str(e)}. Raw content: {content}")
                            # Optionally, display raw content for debugging
                            st.text(content)

            st.success("✅ Done! Hope I saved your time 😊")
        else:
            st.warning("⚠️ Please upload at least one PDF invoice.")

if __name__ == '__main__':
    main()