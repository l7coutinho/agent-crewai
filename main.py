from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Agent
agent_event_locator = Agent(
	role="Pesquisador de eventos esportivos",
	goal="Encontrar eventos esportivos de uma determinada categoria com base nos interesses do usuário.",
	backstory="""
		Você sera um pesquisador de eventos esportivos, com amplo conhecimento em eventos esportivos.
		Sua missão é encontrar eventos esportivos de uma determinada categoria com base nos interesses do usuário
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
		esportivos de uma determinada categoria considerando localização, datas e preferências do usuário.
	""",
	llm=llm
)

# Tasks

task_event_locator = Task(
	description="""
		Encontrar eventos esportivos de uma determinada categoria com base nos interesses do usuário delmitado entre as tags.
		Sua resposta final deve ser uma lista de eventos esportivos com informações relevantes, como nome, local, data e hora e uma breve descricão.

		<evento>
		- Disponibilidade: 16/12/2022 a 18/12/2022
		- Cidade: São Paulo
		- Evento de interesse: Futebol
		</evento>
	""",
	agent=agent_event_locator
)

task_planning_agent = Task(
	description="""
		Criar um itinerário otimizado que maximize a eficiência de deslocamento e minimize o tempo de viagem do usuário.
		Inclua recomendações de transportees e horários de funcionamento.
		A resposta final deve ser um plano de viagem completo, com um cronograma diário e um itinerário otimizado.
	""",
	agent=planning_agent
)
