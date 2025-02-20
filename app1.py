import streamlit as st
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.groq import Groq
from agno.tools.newspaper4k import Newspaper4kTools
import logging

logging.basicConfig(level=logging.DEBUG)

# Streamlit UI
st.title("Startup Idea Analysis Agent 📈")
st.caption("Get the latest trend analysis and startup opportunities based on your topic of interest!")

topic = st.text_input("Enter the Startup Idea of your interest:")
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if st.button("Generate Analysis"):
    if not groq_api_key:
        st.warning("Please enter the required API key.")
    else:
        with st.spinner("Processing your request..."):
            try:
                # Initialize the Groq model
                model = Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key)

                # Define News Collector Agent
                search_tool = DuckDuckGoTools(search=True, news=True, fixed_max_results=5)
                news_collector = Agent(
                    name="News Collector",
                    role="Collects recent news articles on the given topic",
                    tools=[search_tool],
                    model=model,
                    instructions=["Gather latest articles on the topic"],
                    show_tool_calls=True,
                    markdown=True,
                )

                # Define Summary Writer Agent
                news_tool = Newspaper4kTools(read_article=True, include_summary=True)
                summary_writer = Agent(
                    name="Summary Writer",
                    role="Summarizes collected news articles",
                    tools=[news_tool],
                    model=model,
                    instructions=["Provide concise summaries of the articles"],
                    show_tool_calls=True,
                    markdown=True,
                )

                # Define Trend Analyzer Agent
                trend_analyzer = Agent(
                    name="Trend Analyzer",
                    role="Analyzes trends from summaries",
                    model=model,
                    instructions=["Identify emerging trends and startup opportunities based on the topic"],
                    show_tool_calls=True,
                    markdown=True,
                )

                # Step 1: Collect news
                news_response = news_collector.run(f"Collect recent news on {topic}")
                if not news_response or not news_response.content:
                    st.error("Failed to retrieve articles. Check the search tool.")
                    st.stop()
                articles = news_response.content

                # Step 2: Summarize articles
                summary_response = summary_writer.run(f"Summarize the following articles:\n{articles}")
                if not summary_response or not summary_response.content:
                    st.error("Failed to generate summaries. Check the summary writer.")
                    st.stop()
                summaries = summary_response.content

                # Step 3: Analyze trends
                trend_response = trend_analyzer.run(f"Analyze trends from the following summaries:\n{summaries}")
                if not trend_response or not trend_response.content:
                    st.error("Trend analysis failed. Check API response.")
                    st.stop()

                # Display the results
                st.subheader("Trend Analysis and Potential Opportunities on the Startup Idea")
                st.write(trend_response.content)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                logging.error(f"Exception: {e}")
else:
    st.info("Enter the topic and API key, then click 'Generate Analysis' to start.")
