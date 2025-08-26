# AIChatbot


###Document-Based AI Chatbot API###
This project is a backend-only service that provides a text generation API based on the content of a specific document. It uses a Retrieval-Augmented Generation (RAG) approach to find relevant information from the source text and generate a coherent, conversational answer to a user's prompt.

The entire service is built with FastAPI, making it a lightweight and high-performance backend suitable for running on macOS or any other operating system.

Tech Stack
Backend Framework: FastAPI

Web Server: Uvicorn

AI Models:

sentence-transformers/all-MiniLM-L6-v2: For creating vector embeddings (understanding text meaning).

distilgpt2: For generating conversational text.

Core Libraries:

transformers: For accessing pre-trained AI models.

scikit-learn: For efficient similarity search (retrieval).

torch (PyTorch): The deep learning framework that powers the models.

How It Works
The application follows a two-step process to answer a prompt:

Retrieval: When a user sends a prompt (e.g., "Which planet is the largest?"), the system first converts the prompt into a numerical vector. It then uses a scikit-learn similarity search index to find the most relevant text chunks from the source document.

Generation: The retrieved text chunks and the original prompt are combined into a new, detailed prompt. This is then fed to the distilgpt2 generative model, which writes a new, human-like answer based on the context it was given.

This RAG approach ensures that the AI's answers are grounded in the facts provided in the source document.

##Comments on the Results##

The output demonstrates that the system is working as intended. The retrieval step successfully identified the most relevant sentence from the source document that mentions "rings." Subsequently, the generation step correctly extracted the key piece of information—"Saturn"—and provided it as a direct and accurate answer. This confirms that the RAG architecture is effective for building a fact-based chatbot that avoids making up information and stays grounded in the provided context.Still is necessary further improvements in the code for performance improvements.
