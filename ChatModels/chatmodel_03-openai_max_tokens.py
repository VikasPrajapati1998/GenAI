"""
Temperature controls the randomness of a language model's output.

Range Guide:
-------------
Lower Value (0.0 - 0.3) : More deterministic and predictable.
Higher Value (0.7 - 1.5) : More random, creative, and diverse.

Recommended Temperature:
------------------------
- Factual Answers (math, code, facts): 0.0 - 0.3
- Balanced Response (general QA, explanations): 0.5 - 0.7
- Creative writing, story telling, jokes: 0.9 - 1.2
- Maximum randomness (wild ideas, brainstorming): 1.5+

Max_Completion_Tokens controls the length of a language model's output.
You could take any positive integer number in any range.
"""

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI


def main() -> None:
    """Initialize the OpenAI chat model and generate a response."""
    # Load environment variables from .env file
    load_dotenv(find_dotenv())

    # Initialize Chat Model with chosen temperature
    chat_model = ChatOpenAI(
        model="gpt-4o-mini",  # or "gpt-5" if supported in your environment
        temperature=1.1,
        max_completion_tokens = 20,
    )

    # Define the input question
    question: str = "Write a 4 line poem on life."

    # Get the model's response
    response = chat_model.invoke(question)

    # Print the answer content
    print(f"Question: {question}")
    print(f"Answer: {response.content}")


if __name__ == "__main__":
    main()



'''
temperature = 0.0
In the dance of days, we weave our dreams,  
Through laughter and tears, life flows like streams.  
Each moment a treasure, both fleeting and bright,  
In the tapestry of time, we find our light.

temperature = 1.1
Life's a tapestry, woven bright,  
Threads of joy and shadows light,  
In every moment, wisdom grows,
Embrace the journey, as time flows.

max_completion_tokens = 20
In whispers of the morning light,  
Life dances softly, bold yet slight.  
Each moment blooms
'''
