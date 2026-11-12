
# AI-Powered Dataset Understanding & Data Preprocessing Assistant

## Project objective
A Streamlit application that allows users to upload CSV/Excel datasets, understand their structure, detect common data-quality issues, receive preprocessing recommendations, apply selected preprocessing operations, and download the cleaned dataset.

## Current architecture

Upload Dataset
      ↓
Dataset Understanding
      ↓
Data Quality Analysis
      ↓
Recommendation Engine
      ↓
User Selects Preprocessing
      ↓
Python/Pandas/Scikit-learn Processing
      ↓
Cleaned Dataset + Comparison
      ↓
Download CSV

## Technologies
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- OpenPyXL

## Important design decision
The actual calculations and transformations are performed by Python/Pandas/Scikit-learn. The AI layer should be used for recommendations and natural-language explanations rather than blindly changing the data.

## Run locally

1. Install Python.
2. Open a terminal in this project folder.
3. Run:

   pip install -r requirements.txt

4. Configure your Gemini API key (optional, for AI explanations):
   - Add your key to `.env`:
     ```
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

5. Start the application:

   streamlit run app.py

6. Open the local Streamlit URL shown in the terminal.

## Next development stage
Add an LLM layer that receives the calculated dataset profile and recommendation results and generates a concise explanation such as:

- What problems were detected?
- Why was a preprocessing method recommended?
- What are the risks of using that method?
- What should the user verify before applying it?

Do not send the entire dataset to the LLM unnecessarily. Send summarized statistics and detected issues instead.
