import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from app.rag_agent._utils import load_prompt_sections
from app.rag_agent.tools.rag_search import stock_rag_search_tool, news_rag_search_tool
from app.rag_agent.tools.mrktsec_quotes_detail_tool import mrktsec_quotes_detail_tool
from app.rag_agent.tools.technical_price_list_tool import technical_price_list_tool

load_dotenv()

def build_zero_shot_agent(llm):
    tools = [stock_rag_search_tool, news_rag_search_tool, mrktsec_quotes_detail_tool, technical_price_list_tool]
    prompt = load_prompt_sections(os.path.join(os.path.dirname(__file__), "prompts", "prompt_template.txt"))

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        agent_kwargs={
            'prefix': prompt["PREFIX"],
            'format_instructions': prompt["FORMAT_INSTRUCTIONS"],
            'suffix': prompt["SUFFIX"],
        },
        max_iterations=(len(tools) + 4)
    )
    return agent


llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL"), 
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

zero_shot_agent = build_zero_shot_agent(llm)

def run_agentic_rag(prompt):
    try:
        response = zero_shot_agent.invoke({"input": prompt})
        return response["output"]
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Xin lỗi, đã xảy ra lỗi trong quá trình xử lý yêu cầu của bạn."
    

