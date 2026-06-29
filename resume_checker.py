import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import pdfplumber
import os
from dotenv import load_dotenv

# Load Groq API Key
load_dotenv()
groq_apikey = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Resume Checker", page_icon="📄")
st.title("📄 Resume Checker")
st.write("Upload your resume (PDF) and get AI-powered feedback")

if not groq_apikey:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.info("Add `GROQ_API_KEY=your_key_here` in a `.env` file")
else:
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if uploaded_file is not None:
        if st.button("🔍 Evaluate Resume", type="primary"):
            with st.spinner("Analyzing your resume..."):
                try:
                    # Extract text directly from bytes (No tempfile!)
                    with pdfplumber.open(uploaded_file) as pdf:
                        text_pages = [page.extract_text() or "" for page in pdf.pages]
                        context = "\n\n".join(text_pages)

                    if not context.strip():
                        st.warning("Could not extract text from PDF. Try a text-based (not scanned) resume.")
                        st.stop()

                    # Initialize Groq
                    llm = ChatGroq(
                        model="groq/compound-mini",
                        api_key=groq_apikey
                    )

                    # Prompt
                    prompt_template = PromptTemplate(
                        input_variables=["context"],
                        template="""
You are an expert resume reviewer. Evaluate the following resume and provide a detailed score based on clarity, format, ATS compatibility, and skills. Use the following structure for your response:
important instructions:
- Provide a score out of 100.(dont give same score for all resumes)
Score resume standalone (clarity, format, ATS, skills): EXACT structure:

1. **Score**: X/100
2. **Strengths**: 
   • ...(at least three)
3. **Weaknesses**: 
   • ...(at least three)
4. **Skills Mentioned**: 
   • ...(explicitly mentioned)
5. **Recommended Skills**: 
   • ...(additional useful skills)
6. **Next Career Steps**: 
   • ...(suggested next steps)

Resume: {context}
"""
                    )

                    formatted_prompt = prompt_template.format(context=context)

                    # Streaming response
                    st.subheader("📋 Resume Evaluation")
                    response_placeholder = st.empty()
                    full_response = ""

                    for chunk in llm.stream(formatted_prompt):
                        full_response += chunk.content
                        response_placeholder.markdown(full_response)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("👆 Upload your PDF resume to start evaluation.")