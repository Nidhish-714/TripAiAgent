from phi.agent import Agent
# from phi.playground import Playground, serve_playground_app
from phi.tools.calculator import Calculator
from phi.model.groq import Groq
from phi.agent import Agent, RunResponse
from phi.utils.pprint import pprint_run_response
from typing import Iterator, List, Dict, Any
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k
from geopy.geocoders import Nominatim
import time

def get_location_coordinates(location_name: str) -> str:
    """
    Get the GPS coordinates (latitude and longitude) for a specified location.
    
    Args:
        location_name (str): The name of the location to geocode (e.g., "Eiffel Tower, Paris, France")
    
    Returns:
        str: JSON string containing the address, latitude, and longitude of the location
    """
    try:
        # Initialize the geocoder with a custom user agent
        geolocator = Nominatim(user_agent="TravelPlannerAgent")
        
        # Get location information
        location = geolocator.geocode(location_name, exactly_one=True)
        
        # Add a small delay to respect usage limits
        time.sleep(1)
        
        if location:
            result = {
                "address": location.address,
                "latitude": location.latitude,
                "longitude": location.longitude
            }
            return str(result)
        else:
            return f"Coordinates not found for '{location_name}'. Try providing more details like city or country."
    
    except Exception as e:
        return f"Error getting coordinates: {str(e)}. Try another location name or format."

class InteractiveTravelAgent:
    def __init__(self):
        self.agent = Agent(
            model=Groq(
                id="llama-3.3-70b-versatile",
                api_key="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",
                max_tokens=10000
            ),
            markdown=True,
            tools=[
                DuckDuckGo(), 
                Newspaper4k(),
                Calculator(
                    add=True,
                    subtract=True,
                    multiply=True,
                    divide=True,
                    exponentiate=True,
                    factorial=True,
                    is_prime=True,
                    square_root=True,
                ),
                get_location_coordinates
            ],
            description="You are a seasoned travel agent or trip itinerary planner specializing in crafting seamless, personalized travel experiences.",
            instructions=[
                """Your role is to guide the user through an interactive trip planning process with these steps:
                
                1. First, research and suggest popular attractions/places in the requested destination
                2. For each suggested place, use the get_location_coordinates tool to find and include its exact GPS coordinates
                3. Ask the user to select which places they're interested in visiting from your suggestions
                4. Based on their selections, recommend hotels/accommodations in different budget ranges
                5. For each accommodation, also use the get_location_coordinates tool to get its GPS coordinates
                6. Ask the user to select their preferred accommodation
                7. Finally, create a detailed day-by-day itinerary including all selected places, accommodations, transportation options, and budget estimates
                
                At each step, provide relevant information and wait for user input before proceeding to the next step.
                Remember to use search tools to get up-to-date information about attractions, hotels, and other details.
                When calculating budgets, break down costs for accommodation, meals, transportation, and activities.
                
                IMPORTANT: Always use the get_location_coordinates tool to get precise GPS coordinates for every attraction and accommodation you suggest. Make sure to specify the full location name including city/region/country for accurate results.
                """
            ],
            show_tool_calls=True,
            add_datetime_to_instructions=True,
        )
        self.context = {}
        
    def suggest_places(self, destination, duration):
        """Step 1: Suggest places to visit based on destination and duration with coordinates"""
        query = f"""Suggest top attractions and places to visit in {destination} for a {duration} trip.
        
        For each attraction you suggest:
        1. Provide a brief description
        2. Use the get_location_coordinates tool to find its exact GPS coordinates
        3. Number each suggestion for easy reference
        
        Format each attraction with its coordinates clearly visible."""
        
        response_stream = self.agent.run(query, stream=True)
        pprint_run_response(response_stream, markdown=True, show_time=True)
        
        # Store destination and duration in context
        self.context["destination"] = destination
        self.context["duration"] = duration
        
        # After this function runs, collect user input about which places they want to visit
        selected_places = input("\nPlease enter the numbers of the places you want to visit (comma-separated, e.g., 1,3,5): ")
        self.context["selected_places"] = selected_places
        return selected_places
    
    def suggest_accommodations(self):
        """Step 2: Suggest accommodations based on selected places with coordinates"""
        selected_places = self.context.get("selected_places", "")
        destination = self.context.get("destination", "")
        
        query = f"""Based on the user's interest in places {selected_places} in {destination}, suggest accommodation options in different budget ranges (budget, mid-range, luxury) that are conveniently located near these attractions.
        
        For each accommodation:
        1. Provide name, description and approximate price range
        2. Use the get_location_coordinates tool to find its exact GPS coordinates
        3. Mention its proximity to selected attractions
        4. Number each suggestion for easy reference
        
        Format each accommodation with its coordinates clearly visible."""
        
        response_stream = self.agent.run(query, stream=True)
        pprint_run_response(response_stream, markdown=True, show_time=True)
        
        # After this function runs, collect user input about preferred accommodation
        selected_hotel = input("\nPlease enter the number of your preferred accommodation: ")
        self.context["selected_hotel"] = selected_hotel
        return selected_hotel
    
    def create_itinerary(self):
        """Step 3: Create a detailed itinerary based on all selections, including coordinates"""
        destination = self.context.get("destination", "")
        duration = self.context.get("duration", "")
        selected_places = self.context.get("selected_places", "")
        selected_hotel = self.context.get("selected_hotel", "")
        
        query = f"""Create a detailed {duration} itinerary for {destination} including:
        1. Day-by-day schedule visiting the places numbered {selected_places} that the user selected
        2. Accommodation at hotel option {selected_hotel}
        3. Transportation recommendations between attractions
        4. Meal suggestions including local cuisine
        5. Estimated budget breakdown for the entire trip
        6. Include GPS coordinates for every location mentioned in the itinerary (you can reference previously found coordinates)
        
        Organize by day and include estimated times for activities."""
        
        response_stream = self.agent.run(query, stream=True)
        pprint_run_response(response_stream, markdown=True, show_time=True)
        
    def run(self):
        """Run the interactive travel agent workflow"""
        print("Welcome to the Interactive Travel Planner with Precise Location Coordinates!")
        destination = input("Enter your destination: ")
        duration = input("Enter the duration of your trip (e.g., 3 days): ")
        
        # Step 1: Suggest places
        self.suggest_places(destination, duration)
        
        # Step 2: Suggest accommodations based on selected places
        self.suggest_accommodations()
        
        # Step 3: Create detailed itinerary
        self.create_itinerary()
        
        print("\nYour travel planning is complete! Enjoy your trip!")

# Example usage
if __name__ == "__main__":
    travel_agent = InteractiveTravelAgent()
    travel_agent.run()

# To launch as a web app, uncomment these lines:
# playground = Playground(agents=[travel_agent.agent], name="Interactive Travel Planner")
# serve_playground_app(playground, host="0.0.0.0", port=8000)