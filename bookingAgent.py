from phi.agent import Agent
from phi.model.groq import Groq
from phi.utils.pprint import pprint_run_response
from phi.tools.duckduckgo import DuckDuckGo

class TravelOptionsFinder:
    def __init__(self):
        self.agent = Agent(
            model=Groq(
                id="llama-3.3-70b-versatile",
                api_key="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",
                max_tokens=10000
            ),
            markdown=True,
            tools=[DuckDuckGo()],
            description="You are a specialized travel consultant who finds the best transportation and accommodation options for travelers based on their itineraries.",
            instructions=[
                """Your role is to help users find the best travel options and accommodations for their planned trips:
                
                1. Analyze the user's itinerary to understand their travel route and needs
                2. Search for and recommend the best transportation options (flights, trains, buses, etc.) between destinations
                3. Find suitable hotels/accommodations at each stay location with links to booking websites
                4. Provide practical information about transportation between attractions at each destination
                5. Ensure all recommendations include:
                   - Direct links to official booking websites
                   - Approximate price ranges
                   - Key features and benefits
                   - User ratings/reviews when available
                
                Always use the search tool to find current information about transportation options, hotels, and booking platforms.
                Organize your recommendations clearly by destination and date.
                """
            ],
            show_tool_calls=True,
            add_datetime_to_instructions=True,
        )
        self.context = {}
        
    def find_transportation_options(self, itinerary):
        """Find best transportation options between destinations in the itinerary"""
        query = f"""Based on the following itinerary, search for and recommend the best transportation options (flights, trains, buses, etc.) between each destination:

        {itinerary}
        
        For each leg of the journey:
        1. Search for available transportation options (flight routes, train services, bus lines)
        2. Find the websites where these can be booked
        3. Include approximate costs, duration, and schedule information
        4. Highlight the most convenient or cost-effective options
        
        Format your response by journey leg (e.g., "City A to City B") and include direct links to booking websites."""
        
        response_stream = self.agent.run(query, stream=True)
        pprint_run_response(response_stream, markdown=True, show_time=True)
        
        # Store itinerary in context
        self.context["itinerary"] = itinerary
        
        return self.context
    
    def find_accommodation_options(self):
        """Find accommodation options for each destination in the itinerary"""
        itinerary = self.context.get("itinerary", "")
        
        query = f"""Based on the following itinerary, search for and recommend accommodation options at each destination:

        {itinerary}
        
        For each destination where the traveler will stay overnight:
        1. Search for hotel options in different price ranges (budget, mid-range, luxury)
        2. Find the official websites or booking platforms where these accommodations can be reserved
        3. Include approximate nightly rates, key amenities, and location advantages
        4. Note any special deals or promotions currently available
        
        Format your response by destination and include direct links to booking websites for each recommended accommodation."""
        
        response_stream = self.agent.run(query, stream=True)
        pprint_run_response(response_stream, markdown=True, show_time=True)
        
        return self.context
    
    def find_local_transportation(self):
        """Find local transportation options within each destination"""
        itinerary = self.context.get("itinerary", "")
        
        query = f"""Based on the following itinerary, search for and recommend local transportation options within each destination:

        {itinerary}
        
        For each destination:
        1. Search for public transportation options (metro, bus, tram, etc.)
        2. Find information about ride-sharing services or taxis
        3. Look for any transportation passes or cards that might save money
        4. Include websites where tickets or passes can be purchased in advance
        5. Mention transportation options between key attractions mentioned in the itinerary
        
        Format your response by destination and include direct links to official transportation websites or apps."""
        
        response_stream = self.agent.run(query, stream=True)
        pprint_run_response(response_stream, markdown=True, show_time=True)
        
        return self.context
    
    def create_comprehensive_plan(self):
        """Create a comprehensive travel and booking plan"""
        itinerary = self.context.get("itinerary", "")
        
        query = f"""Create a comprehensive travel and booking plan based on this itinerary:

        {itinerary}
        
        Include:
        1. A complete day-by-day breakdown with all transportation and accommodation recommendations
        2. Direct booking links for each recommended service
        3. A suggested booking timeline (which bookings should be made first)
        4. Estimated total budget for transportation and accommodations
        5. Tips for getting the best deals on the recommended services
        
        Format this as a complete travel booking guide that the traveler can follow step by step."""
        
        response_stream = self.agent.run(query, stream=True)
        pprint_run_response(response_stream, markdown=True, show_time=True)
    
    def run(self):
        """Run the travel options finder workflow"""
        print("Welcome to the Travel Options and Booking Finder!")
        print("Please enter your travel itinerary with destinations, dates, and key attractions:")
        itinerary = """
Trip to Japan: May 15-25, 2025
- Tokyo (May 15-18): Visit Tokyo Tower, Senso-ji Temple, Shibuya Crossing
- Kyoto (May 19-22): Visit Fushimi Inari Shrine, Kinkaku-ji, Arashiyama Bamboo Grove
- Osaka (May 23-25): Visit Osaka Castle, Dotonbori, Universal Studios Japan
"""
        
        # Step 1: Find transportation options between destinations
        self.find_transportation_options(itinerary)
        
        # Step 2: Find accommodation options
        self.find_accommodation_options()
        
        # Step 3: Find local transportation options
        self.find_local_transportation()
        
        # Step 4: Create comprehensive plan
        self.create_comprehensive_plan()
        
        print("\nYour travel booking plan is complete!")

# Example usage
if __name__ == "__main__":
    options_finder = TravelOptionsFinder()
    options_finder.run()

# Example input itinerary:
"""
Trip to Japan: May 15-25, 2025
- Tokyo (May 15-18): Visit Tokyo Tower, Senso-ji Temple, Shibuya Crossing
- Kyoto (May 19-22): Visit Fushimi Inari Shrine, Kinkaku-ji, Arashiyama Bamboo Grove
- Osaka (May 23-25): Visit Osaka Castle, Dotonbori, Universal Studios Japan
"""