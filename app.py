import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

def initialize_groq_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        max_tokens=2000
    )

def generate_response(use_case, input_text, additional_input=None):
    llm = initialize_groq_llm()
    templates = {
    "Summarization": "Summarize the following text into a clear and concise paragraph, capturing the most important details: \"{input_text}\"",
    "Content Generation": "Create a {content_type} about \"{input_text}\" in a {style} style. Ensure it is engaging, well-structured, and suitable for the intended audience.",
    "Code Generation": "Write a {language} function to {task_description}. Include meaningful comments to explain the code for better understanding.",
    "Sentiment Analysis": "Determine the sentiment of the following text: \"{input_text}\" and respond with Positive, Negative, or Neutral. Provide a brief explanation for your assessment.",
    "Translation": "Translate the following text from {source_language} to {target_language}. Ensure accuracy and preserve the cultural nuances: \"{input_text}\"",
    "Question Answering": "As an expert in {domain}, answer the following question concisely and clearly: \"{input_text}\". Ensure your response is accurate and directly addresses the question.",
    "Personalized Recommendations": "Based on the preference \"{user_preferences}\", suggest {recommendation_type}. Provide reasons why these suggestions are suitable.",
    "Educational Applications": "Explain the concept of \"{input_text}\" in simple terms, suitable for a {level} level student. Use relatable examples to enhance understanding.",
    "Opinion and Analysis": "Critically analyze the following statement: \"{input_text}\". Discuss its strengths, weaknesses, and overall implications.",
    "Creative Writing Assistance": "Continue this poem with the next two lines, maintaining the original tone and rhythm: \"{input_text}\"",
    "Medical or Technical Advice": "As a {domain_expert}, provide a detailed explanation of \"{input_text}\". Include key points, practical examples, and actionable advice where applicable.",
    "Conversational Agents": "Respond to the user in a friendly, engaging, and empathetic tone: \"{input_text}\". Make the response helpful and contextually appropriate.",
    }


    template = templates.get(use_case)

    if not template:
        return "Invalid use case selected."

    prompt_inputs = {"input_text": input_text}
    if use_case in [
        "Content Generation", "Code Generation", "Translation",
        "Question Answering", "Personalized Recommendations",
        "Educational Applications", "Medical or Technical Advice"
    ]:
        prompt_inputs.update(additional_input)

    prompt = PromptTemplate(input_variables=list(prompt_inputs.keys()), template=template)
    formatted_prompt = prompt.format(**prompt_inputs)
    response = llm.invoke(formatted_prompt)

    return response.content

st.set_page_config(page_title="Chat With AI", page_icon="🤖", layout="centered")
st.header("Chat With AI 🤖")

use_case = st.selectbox(
    "Select a use case",
    [
        "Summarization",
        "Content Generation",
        "Code Generation",
        "Sentiment Analysis",
        "Translation",
        "Question Answering",
        "Personalized Recommendations",
        "Educational Applications",
        "Opinion and Analysis",
        "Creative Writing Assistance",
        "Medical or Technical Advice",
        "Conversational Agents",
    ]
)

input_text = st.text_area("Enter your input text or question")

additional_input = {}

if use_case == "Content Generation":
    additional_input["content_type"] = st.text_input("Enter the content type (e.g., Story, Report, etc.)")
    additional_input["style"] = st.text_input("Enter the writing style (e.g., Formal, Informal)")

elif use_case == "Code Generation":
    additional_input["language"] = st.text_input("Enter the programming language")
    additional_input["task_description"] = st.text_input("Describe the task the code should perform")

elif use_case == "Translation":
    additional_input["source_language"] = st.text_input("Enter the source language")
    additional_input["target_language"] = st.text_input("Enter the target language")

elif use_case == "Question Answering":
    additional_input["domain"] = st.selectbox("Select the domain", ["General Knowledge", "Technology", "Science", "Health", "Business", "Engineering"])

elif use_case == "Personalized Recommendations":
    additional_input["recommendation_type"] = st.text_input("Enter the type of recommendation (e.g., Movies, Books)")
    additional_input["user_preferences"] = st.text_input("Enter user preferences")

elif use_case == "Educational Applications":
    additional_input["level"] = st.selectbox("Select the student level", ["Beginner", "Intermediate", "Advanced"])

elif use_case == "Medical or Technical Advice":
    additional_input["domain_expert"] = st.text_input("Enter the type of expert (e.g., Doctor, Engineer)")

if st.button("Generate Response"):
    if not input_text.strip():
        st.error("Please enter valid input text.")
    else:
        response = generate_response(use_case, input_text, additional_input)
        st.write(response)