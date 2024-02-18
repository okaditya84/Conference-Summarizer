
# Conference Summarizer AI
## Overview
The Conference Summarizer AI is an intelligent system designed to facilitate the summarization of conferences, meetings, lectures, and expert talks. It allows speakers to upload initial materials in various formats such as PDF, PPT, and Docx. The system enables the audience to ask questions through text or voice inputs, generating conversational responses with context. Deployed on Streamlit Cloud, it utilizes the Gemini API and Google Speech Recognition APIs for seamless processing.

## Features
- Upload Initial Materials: Speakers can upload knowledge materials in PDF, PPT, and Docx formats.
- Conversational Interaction: Audience can ask questions via text or voice, receiving contextualized responses.
- Downloadable Summaries: Users can download summarized content to their mobile devices and computers.
- Integration with Google Docs: Plan to connect the system with Google Docs to provide uniform URL links to summary documents.
- Feedback Mechanism: Users can provide feedback to improve the system.
- Knowledge Base Enhancement: Append user questions and answers to the knowledge base for richer context.

## Try it live
Check out the live demo: https://conferenceai.streamlit.app/


## Installation
To run the code in this repository, you need to install the necessary dependencies. Use the following command to install the required packages:

```bash
  pip install transformers[torch] datasets accelerate -U

```
### Usage

Clone this repository:

```bash
git clone https://github.com/okaditya84/Conference-Summarizer
cd Conference-Summarizer
```

Make a python environment and activate:
```bash
python -m venv env
env\Scripts\activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

Get your Google API Key and store in the environment variable.

Run the provided code using a Python environment:
```bash
streamlit run app.py
```

## Usage
Follow these simple steps to utilize the Conference Summarizer AI:

#### Upload Knowledge Base Material:

- Navigate to the "Upload" section on the web interface.
- Click on the "Choose File" button to select the knowledge base material in PDF, PPT, or Docx format.
- After selecting the file, click on the "Submit and Process" button to initiate the processing of the material.
#### Ask Questions:

- Type your question in the text box labeled "Type your question here" or enable the "Voice Input" checkbox.
- If using voice input, a small contextual description is required. Type this context in the same text box.
- Hit Enter or click the "Submit Voice Input" button.

#### Generate Answers:

- The system will process the input and generate contextual answers based on the uploaded knowledge base.
- Review the answers provided in the chat or conversation section.

#### Download Summary:

- Once satisfied with the responses, click on the "Generate Summary" button.
- The system will compile a summary of the conversation, and you can download it to your device.

#### Clear History:

- To start a new conversation or session, click on the "Clear History" button.
- This action resets the chat history, allowing you to begin a fresh interaction.


## Future scopes
- Improve UI/UX design for a more user-friendly experience.
- Implement Google Docs integration for easier access to summary documents.
- Continuous improvement based on user feedback.
## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://adityajethani.vercel.app/)

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adityajethani/)

[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/okaditya84)

[![Github](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=github&logoColor=white)](https://github.com/okaditya84)

