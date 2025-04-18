import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# --- UI Inputs ---
st.set_page_config(page_title="🧳 AI Trip Planner", layout="centered")

st.title("🧳 AI-Powered Trip Planner with CrewAI")
st.markdown("Get a personalized, budget-friendly itinerary with real-time travel info!")

# User Inputs
destination = st.text_input("Enter your destination:", "Hyderabad")
budget = st.number_input("Enter your budget (INR):", min_value=100, max_value=100000, value=5000, step=100)

# API Keys
gpt_key = st.text_input("Enter your GPT API Key", type="password")
serper_key = st.text_input("Enter your Serper API Key", type="password")

# Run Button
run_trip_planner = st.button("🧭 Plan My Trip")

if run_trip_planner:
    if not gpt_key or not serper_key:
        st.warning("Please provide both GPT and Serper API keys.")
    else:
        os.environ["GPT_API_KEY"] = gpt_key
        os.environ["SERPER_API_KEY"] = serper_key

        # Initialize LLM & Tool
        search_tool = SerperDevTool()
        llm = LLM(
            model="gpt-4",  # Or gemini/gemini-1.5-flash
            verbose=True,
            temperature=0.5,
            api_key=gpt_key
        )

        # Define Agents
        researcher = Agent(
            role="Travel Researcher",
            goal=f"Find historical sites, public transport, hotels, and real-time weather for {destination}.",
            verbose=True,
            memory=True,
            backstory="You are an expert travel researcher, providing up-to-date information about history-focused trips.",
            llm=llm,
            tools=[search_tool],
            allow_delegation=True
        )

        budget_planner = Agent(
            role="Budget Planner",
            goal=f"Find budget buses, cabs, and activities within {budget} for {destination}.",
            verbose=True,
            memory=True,
            backstory="You are a skilled budget analyst ensuring trips fit within financial constraints.",
            llm=llm,
            tools=[search_tool],
            allow_delegation=True
        )

        itinerary_planner = Agent(
            role="Itinerary Planner",
            goal=f"Create a 1-day itinerary for {destination}, ensuring all historical sites are covered under {budget}.",
            verbose=True,
            memory=True,
            backstory="You are an expert in trip planning, ensuring travelers get the best experience within their budget.",
            llm=llm,
            tools=[search_tool],
            allow_delegation=False
        )

        # Tasks
        research_task = Task(
            description=f"Find the best historical sites, weather forecast, and public transport for {destination}.",
            expected_output="A list of top historical sites, a real-time weather update.",
            tools=[search_tool],
            agent=researcher
        )

        budget_task = Task(
            description=f"Find budget buses, cabs and food/transport costs for {destination}. Ensure total cost stays under {budget}.",
            expected_output="A full cost breakdown (buses, cabs, food, attractions) ensuring the budget is maintained.",
            tools=[search_tool],
            agent=budget_planner
        )

        itinerary_task = Task(
            description=f"Plan a 1-day itinerary for {destination}, focusing on historical sites, budget constraints, and real-time weather conditions.",
            expected_output="A detailed 1-day plan, considering weather and budget constraints, with transport recommendations.",
            tools=[search_tool],
            agent=itinerary_planner
        )

        # Run the Crew
        crew = Crew(
            agents=[researcher, budget_planner, itinerary_planner],
            tasks=[research_task, budget_task, itinerary_task],
            process=Process.sequential
        )

        with st.spinner("Planning your trip..."):
            result = crew.kickoff(inputs={"destination": destination, "budget": str(budget)})
        
        # Show Result
        st.success("🎉 Trip Plan Ready!")
        st.markdown("### 🗓️ Your 1-Day Itinerary")
        st.write(result)
