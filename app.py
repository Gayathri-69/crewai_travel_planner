import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# Set static API keys
os.environ["GPT_API_KEY"] = "AIzaSyA78loTXEYfy4SAiEFi7rRG8vy0gHRSFuM"
os.environ["SERPER_API_KEY"] = "e77b760ad536892826a6a56e13eeb94eb7dc8495"

# --- Streamlit UI Setup ---
st.set_page_config(page_title="🧳 AI Trip Planner", layout="centered")
st.title("🧳 AI-Powered Trip Planner")
st.markdown("Plan a 1-day historical trip with real-time data and a budget!")

# --- Manual User Inputs ---
destination = st.text_input("🌍 Enter your destination:", placeholder="e.g., Hyderabad")
budget = st.number_input("💸 Enter your budget (INR):", min_value=1000, max_value=50000, value=5000, step=500)

run = st.button("🚀 Plan My Trip")

if run:
    if not destination.strip():
        st.warning("Please enter a destination.")
    else:
        # Initialize tools
        search_tool = SerperDevTool()
        llm = LLM(
            model="gemini/gemini-1.5-flash"
            verbose=True,
            temperature=0.5,
            api_key=os.environ["GPT_API_KEY"]
        )

        # Agents
        researcher = Agent(
            role="Travel Researcher",
            goal=f"Find historical sites, transport, hotels, and weather for {destination}.",
            verbose=True,
            memory=True,
            backstory="Expert in historic destinations and weather updates.",
            llm=llm,
            tools=[search_tool],
            allow_delegation=True
        )

        budget_planner = Agent(
            role="Budget Planner",
            goal=f"Suggest cheap transport and attractions under ₹{budget} for {destination}.",
            verbose=True,
            memory=True,
            backstory="Keeps the trip within budget.",
            llm=llm,
            tools=[search_tool],
            allow_delegation=True
        )

        itinerary_planner = Agent(
            role="Itinerary Planner",
            goal=f"Plan a 1-day trip in {destination} with historical sites and budget ₹{budget}.",
            verbose=True,
            memory=True,
            backstory="Creates practical and interesting itineraries.",
            llm=llm,
            tools=[search_tool],
            allow_delegation=False
        )

        # Tasks
        research_task = Task(
            description=f"Find historical sites, weather, and public transport for {destination}.",
            expected_output="Top historical sites, weather forecast, and transport options.",
            tools=[search_tool],
            agent=researcher
        )

        budget_task = Task(
            description=f"List cheap travel, food, and activity costs in {destination} under ₹{budget}.",
            expected_output="Budget breakdown for travel, food, and entry tickets.",
            tools=[search_tool],
            agent=budget_planner
        )

        itinerary_task = Task(
            description=f"Create a 1-day itinerary in {destination} with weather, budget, and site visits.",
            expected_output="Detailed 1-day itinerary with estimated costs.",
            tools=[search_tool],
            agent=itinerary_planner
        )

        # Crew
        crew = Crew(
            agents=[researcher, budget_planner, itinerary_planner],
            tasks=[research_task, budget_task, itinerary_task],
            process=Process.sequential
        )

        with st.spinner("⏳ Planning your trip..."):
            result = crew.kickoff(inputs={"destination": destination, "budget": str(budget)})

        st.success("🎉 Trip Plan Ready!")
        st.markdown("### 📋 Your Itinerary")
        st.write(result)
