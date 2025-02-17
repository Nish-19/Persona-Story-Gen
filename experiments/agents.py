"""
define custom agents in the form of a class
"""

import os
from openai import AzureOpenAI
from swarm import Swarm, Agent


class StoryAgent:
    """
    Collection of LLMs that can generate stories
    """

    def __init__(self):
        # NOTE: Load the instructions
        pass
