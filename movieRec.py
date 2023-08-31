
from typing import List, Union, Optional, Tuple
import re
from enum import Enum

from langchain import SerpAPIWrapper, LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, BaseSingleActionAgent,AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain.prompts import BaseChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackManager
from pinecon import create_index, get_movie_recommendations
from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field
import pinecone


movie_index = 'movies-index'
from AppLogger import get_logger
logger = get_logger("index")

agent_executor = None

"""
a function that defines how the agent plans its actions based on the intermediate steps and other arguments:
"""

class MyAgent(LLMSingleActionAgent):
    def plan(self, intermediate_steps, **kwargs):
        if len(intermediate_steps) == 0:
            return AgentAction(tool="Genre Transformer", tool_input=kwargs["input"], log="")
        else:
            genre = intermediate_steps[-1][1]
            return AgentAction(tool="Search", tool_input=genre, log="")


# Define your list of genres
genres = [ "Action",
  "Adventure",
  "Animation",
  "Children",
  "Comedy",
  "Crime",
  "Documentary",
  "Drama",
  "Fantasy",
  "Film-Noir",
  "Horror",
  "Musical",
  "Mystery",
  "Romance",
  "Sci-Fi",
  "Thriller",
  "War",
  "Western",
]

def transform_input_to_genre(user_input):
    global genres

    best_match = None
    highest_similarity = 0

    for genre in genres:
        similarity = fuzz.ratio(user_input.content, genre.lower())
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = genre
    return best_match


tools = [ Tool.from_function(
    func=transform_input_to_genre,
    name="Genre Transformer",
    description="Transforms any input into a movie genre"
)]


llm_template = """
Answer the following questions as best you can, but speaking as a pirate might speak.
You have access to the following tools:
{tool_names}
Use the following format:
Question: {input}
Thought: you should always think about what to do
{intermediate_steps}
Thought: I now know the final answer
Final Answer: 
"""

class featureRecTemplate(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]

    def format_messages(self, **kwargs):

        intermediate_steps = kwargs['intermediate_steps']
        kwargs['genres'] = genres

        kwargs['tools'] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])

        thoughts = ''
        for action, observation  in intermediate_steps:
            thoughts += action.tool_input.content
            thoughts += f"\nObservation: {observation}\nThought:"

        kwargs['agent_scratchpad'] = thoughts
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content = formatted)]


class CustomOutputParser(AgentOutputParser):       

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        if "Final Answer" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-2].strip()},
                log=llm_output,
            )

        regex = r"Action:\s*(.*?)\n*Action Input:\s*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

prompt = featureRecTemplate(
    template = llm_template,
    tools = tools,
    input_variables=["input", "intermediate_steps"]
)

output_parser = CustomOutputParser()

llm = ChatOpenAI(temperature=0)

llm_chain = LLMChain(llm=llm, prompt=prompt)       

tool_names = [tool.name for tool in tools]


# Define your custom class as a subclass of AgentExecutor and BaseModel
class CustomAgentExecutor(AgentExecutor, BaseModel):
    # Define your custom fields and types using Field
    intermediate_steps: List[Tuple[AgentAction, str]] = Field(default_factory=list)
    observation: str = Field(default="")

    def __init__(self, agent: Union[LLMSingleActionAgent, BaseSingleActionAgent], tools: List[Tool], callback_manager: Optional[BaseCallbackManager] = None,
            verbose: bool = False):

        super().__init__(agent=agent, tools=tools, llm_chain = agent.llm_chain, output_parser = agent.output_parser,
            callback_manager=callback_manager, verbose=verbose)

    def run(self, user_input: HumanMessage) -> str:
        # Loop until agent finishes
        while True:
            # Check if there are any intermediate steps
            if len(self.intermediate_steps) == 0:
                # Execute the tool specified by the agent action and get the observation
                action = self.agent.plan(self.intermediate_steps, input=user_input)
                self.observation = self.execute_tool(action)
                # Append the action and observation to the intermediate steps
                self.intermediate_steps.append((action, self.observation))
            else:
                # Pass the user input and intermediate steps to the language model and get its output
                output = self.run_llm(user_input)
                # Check if output is an agent action or an agent finish
                if isinstance(output, AgentAction):
                    # Execute the tool specified by the agent action and get the observation
                    action = output
                    self.observation = self.execute_tool(action)
                    # Append the action and observation to the intermediate steps
                    self.intermediate_steps.append((action, self.observation))
                elif isinstance(output, AgentFinish):
                    # Return the final answer as a string
                    return output.return_values["output"]
                else:
                    # Raise an error if output is neither an agent action nor an agent finish
                    raise ValueError(f"Invalid output type: {type(output)}")

    def execute_tool(self, action: AgentAction) -> str:
        # Find the tool by name
        for ttool in self.tools:
            if ttool.name == action.tool:
                tool = ttool
                break
        # Execute the tool and get the observation
        observation = tool.func(action.tool_input)
        # Add the stop signal to the observation
        observation += "\nFinal Answer:"
        # Return the observation
        return observation

    def run_llm(self, user_input: HumanMessage) -> Union[AgentAction, AgentFinish]:
        # Pass user input and intermediate steps to language model
        llm_output = self.agent.llm_chain.run(input=user_input, intermediate_steps=self.intermediate_steps)
        # Add observation to language model output
        llm_output += f"\nObservation: {self.observation}"
        # Pass language model output to output parser
        parsed_output = self.agent.output_parser.parse(llm_output)
        # Check if parsed output is empty or invalid
        if parsed_output == "" or not isinstance(parsed_output, (AgentAction, AgentFinish)):
            # Raise an error if parsed output is empty or invalid
            raise ValueError(f"Invalid or empty output from language model: {parsed_output}")
        else:
            # Return parsed output otherwise
            return parsed_output

def createAgent():
    global agent_executor
    agent = MyAgent(llm_chain = llm_chain, output_parser = output_parser, stop=["Final Answer:"], allowed_tools = tool_names)
    agent_executor = CustomAgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    create_index("movies.csv", movie_index)

def shutdown_index():
    pinecone.delete_index(movie_index)

def runAgent(userInput, number):
    global agent_executor

    uMsg = HumanMessage(content=userInput.lower(), metadata={"language": "en", "sender": "user"})
    output = agent_executor.run(uMsg)
    pgenre = output.split("Observation: ")[1]
    return pgenre, get_movie_recommendations(pgenre, number)

