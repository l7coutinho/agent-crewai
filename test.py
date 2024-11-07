from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai_tools import (
    SerperDevTool,
    WebsiteSearchTool
)

# Load environment variables
load_dotenv()

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=4000)

# Tools
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool(website="https://www.sympla.com.br/")

def create_event_locator_agent():
    """
    Cria um agente de localização de eventos com o objetivo especificado.
    
    Returns:
    Agent: O agente de localização de eventos.
    """
    agent_event_locator = Agent(
        role=f"Pesquisador de eventos musicais.",
        goal=f"Encontrar eventos de música em todo Brasil.",
        backstory=f"""
            Você sera um pesquisador de eventos, com amplo conhecimento em eventos musicais de diversos generos.
            Sua missão é encontrar eventos de música em todo o Brasil.
        """,
        tools=[search_tool, web_rag_tool],
        llm=llm,
        memory=False
    )
    
    return agent_event_locator

def create_event_locator_task(goal, local, date):
    """
    Cria uma tarefa de localização de eventos com o objetivo especificado.
    
    Parameters:
    goal (str): O objetivo da pesquisa de eventos.
    
    Returns:
    Task: A tarefa de localização de eventos.
    """
    task_event_locator = Task(
        description=f"""
            Encontrar eventos sobre {goal}, no {local} no periodo {date}.
            Sua resposta final deve ser uma lista com todos os eventos musicais levando em consideração apenas as informações contidas no site,
            contendo informações relevantes, como nome, data, local e link.
        """,
        expected_output="Lista de eventos contendo: Nome do evento, data do evento e Local do evento e link do evento",
        agent=create_event_locator_agent()
    )
    
    return task_event_locator

# Crew
def run_event_locator(goal, local, date):
    crew = Crew(
        agents=[create_event_locator_agent()],
        tasks=[create_event_locator_task(goal, local, date)],
        verbose=True
    )

    result_output = crew.kickoff()
    return result_output

result_output = run_event_locator("pagode e samba", "Rio de Janeiro", "novembro de 2024")

print(result_output)