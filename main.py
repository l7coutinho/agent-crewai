from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Agent
agent_event_locator = Agent(
    role="Pesquisador de eventos esportivos",
    goal="Encontrar eventos esportivos de uma determinada categoria com base nos interesses do usuário.",
    backstory="""
        Você sera um pesquisador de eventos esportivos, com amplo conhecimento em eventos esportivos.
        Sua missão é encontrar eventos esportivos de uma determinada categoria com base nos interesses do usuário
    """,
    llm=llm
)

planning_agent = Agent(
    role="Planejador de itinerário",
    goal="""
        Criar um itinerário otimizado que maximize a eficiência de deslocamento e minimize o
        tempo de viagem, considerando preferências do usuário, horários de funcionamento dos
        locais e opções de transporte disponíveis.
    """,
    backstory="""
        Com sua habilidade em planejar eventos esportivos,
        você será capaz de criar um itinerário otimizado para encontrar eventos
        esportivos de uma determinada categoria considerando localização, datas e preferências do usuário.
    """,
    llm=llm
)

# Tasks
task_event_locator = Task(
    description="""
        Encontrar eventos esportivos de uma determinada categoria com base nos interesses do usuário delimitado entre as tags.
        Sua resposta final deve ser uma lista de eventos esportivos com informações relevantes, como nome, local, data e hora e uma breve descricão.

        <evento>
        - Disponibilidade: 20/10/2024 a 24/10/2024
        - País: Brasil
        - Evento de interesse: Futebol
        </evento>
    """,
    expected_output="""Lista de eventos esportivos contendo:
        - Nome do evento
        - Local
        - Data e hora
        - Breve descrição
    """,
    agent=agent_event_locator
)

task_planning_agent = Task(
    description="""
        Criar um itinerário otimizado que maximize a eficiência de deslocamento e minimize o tempo de viagem do usuário.
        Inclua recomendações de transportees e horários de funcionamento.
        A resposta final deve ser um plano de viagem completo, com um cronograma diário e um itinerário otimizado.
    """,
    expected_output="""Plano de viagem detalhado contendo:
        - Cronograma diário
        - Itinerário otimizado
        - Recomendações de transporte
        - Horários de funcionamento
    """,
    agent=planning_agent
)

# Crew
crew = Crew(
    agents=[agent_event_locator, planning_agent],
    tasks=[task_event_locator, task_planning_agent],
    verbose=True
)

result_output = crew.kickoff()
 