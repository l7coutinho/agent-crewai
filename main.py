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
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=4000)

# Tools
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

# Agent
agent_event_locator = Agent(
    role="Pesquisador de eventos {goal}.",
    goal="Encontrar eventos musicais em todo Brasil.",
    backstory="""
        Você sera um pesquisador de eventos, com amplo conhecimento em eventos {goal} de diversos generos.
        Sua missão é encontrar eventos {goal} em todo o Brasil.
    """,
    tools=[search_tool, web_rag_tool],
    llm=llm,
    memory=False
)

# planning_agent = Agent(
#     role="Planejador de itinerário",
#     goal="""
#         Criar um itinerário otimizado que maximize a eficiência de deslocamento e minimize o
#         tempo de viagem do usuário.
#     """,
#     backstory="""
#         Com sua habilidade em planejar eventos,
#         você será capaz de criar um itinerário otimizado que maximize a eficiência de deslocamento e
#         minimize o tempo de viagem do usuário.
#     """,
#     llm=llm,
#     memory=False
# )

# Tasks
task_event_locator = Task(
    description="""
        Encontrar eventos sobre o tema {goal} em todo brasil no mês de novembro de 2024.
        Sua resposta final deve ser uma lista de eventos levando em consideração apenas as informações contidas no site,
		contendo informações relevantes, como nome, data e local.
    """,
    expected_output="Lista de eventos contendo: Nome do evento, data do evento e Local do evento",
    agent=agent_event_locator
)

# task_planning_agent = Task(
#     description="""
#         Criar um itinerário otimizado que maximize a eficiência de deslocamento e minimize o tempo de viagem do usuário.
#         Inclua recomendações de transportes e horários de funcionamento.
#     """,
#     expected_output="Itinerário contendo sugestãos de transporte e horários de funcionamento.",
#     agent=planning_agent
# )

# Crew
crew = Crew(
    agents=[agent_event_locator],
    tasks=[task_event_locator],
    verbose=True
)


result_output = crew.kickoff()
 