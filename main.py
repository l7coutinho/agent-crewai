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


