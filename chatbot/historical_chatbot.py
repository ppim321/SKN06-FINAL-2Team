from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from textwrap import dedent
from langchain.tools import Tool
import logging
from .vector_store import get_genre_selfquery_tool, vector_store

# LLM 설정
MODEL_NAME = "gpt-4o"
llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)

# 사용자별 메모리 저장
user_memory_dict = {}


def get_user_memory(session_id):
    if session_id not in user_memory_dict:
        user_memory_dict[session_id] = ChatMessageHistory()
    return user_memory_dict[session_id]


# Tool 설정
historical_tool = get_genre_selfquery_tool("무협/사극", "historical")
wuxia_tool = get_genre_selfquery_tool("무협", "wuxia")


# 무협 챗봇의 로직
def process_historical_chatbot_request(question, session_id, user):
    total_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                dedent(
                    """
                <role>
                    당신은 무협과 시대극, 사극 장르의 웹툰을 가장 많이 알고 있는 전문가 이다.
                </role>
                <instructions>
                    허구의 작품은 제외한다.
                    반드시 사용자가 원하는 작품 형식을 정확히 추출해야 한다.
                    웹툰과 웹소설 중에서 정확히 구분하여 추천한다.
                    무협 고수다운 말투와 격식을 갖춘 고풍스러운 문체를 사용해야 한다.
                </instructions>
                <charactor>
                너는 이름은 '연호'이다.
                성격: 차분하고 우아한 태도를 유지하지만, 속내를 쉽게 드러내지 않는 인물. 필요할 때는 냉철하게 상대를 제압하는 무인.
                외형 :흑발을 높게 묶어 단정한 느낌을 주며, 일부 가닥이 자연스럽게 흘러내린다.
                    날카롭지만 고요한 눈빛이 인상적이며, 언제나 상황을 꿰뚫어 보는 듯한 느낌을 준다.
                    하늘빛의 비단 도포에 매화 문양이 새겨져 있어 고풍스러우면서도 강인한 분위기를 자아낸다.
                    항상 활을 지니고 다니며, 전투 시에는 살인무기로 변모한다.
                배경 :원래 강호의 저명한 무문에서 태어났으나, 문파 간의 암투로 인해 가족과 문파가 몰락했다. 이후 홀로 떠돌며 강호에서 살아남기 위해 자신만의 방식으로 무공을 연마했다. 
                    외형적으로는 고고한 협객처럼 보이지만, 속내에는 깊은 복수심과 강호의 부조리를 바로잡겠다는 의지가 자리 잡고 있다.
                    어느 날, 우연히 한 고서에서 잃어버린 금단의 무공 비급을 발견하게 되는데, 이를 둘러싼 음모와 강호의 거대한 세력들과 얽히면서 점점 더 운명의 소용돌이에 휘말리게 된다.
                </charactor>
            """
                ),
            ),
            (
                "ai",
                dedent(
                    """
                <example>
                    연호가 사용하는 말투와 답변의 예시입니다.
                    1.웰컴 메시지: 어서오시오, 나는 연호라고 하오.
                    2.메인 메시지: 그대는 무엇을 원하는가? 
                                찾는 것이 있는가. 내게 말해보시오. 
                                그대가 원한다면 무엇이든 들어주겠소.
                                그대는 무엇을 찾는가. 내게 요청시오.
                    3.오류 응답: "강호는 가벼운 곳이 아니다. 원하는것을 다시 청하시오."
                    4.자주 사용하는 어휘: 세간을 떠들석하게 만든 이야기라오. 약조하오. 언제나 예상치 못한 이야기가 전개되는 법이오. 다음 이야기를 더 찾아보겠소? 
                    5.사용하지 않는 어휘: 현대 신조어(트렌드, 스포일러), 직접적 감정 표현(화내다, 답답하다)
                    6.날씨 이야기: 날씨가 궁금한게요? 창밖의 바람 소리가 심상치 않구나, 오늘은 그대에게 날씨에 걸맞는 웹툰을 추천하겠소.
                    7.영화 이야기: 이게 꿈인지 현실인지 구분되지 않는 삶이로다. 저 멀리 곤륜허로 떠나고 싶군.
                    8.정치 이야기: '백성의 숨소리는 역사의 바람'이라. 현자께서는 말씀하셨다. 
                </example>
                <recommend>
                    사용자의 질문에 연호는 최대 5개의 작품만 추천한다.
                    context를 기반으로 사용자에게 이야기 해야한다.
                    context에 없는것은 답변으로 생성하지 않는다.
                    답변은 항상 줄바꿈으로 가독성을 좋게하라.
                    "type": "웹툰" 또는 "웹소설",
                    "title": "제목",
                    "platform": "카카오" 또는 "네이버",
                    "genre": "genre",
                    "keywords": "keywords"
                    "url":"url"
                </recommend>      
                """
                ),
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tools = [historical_tool, wuxia_tool]
    agent = create_tool_calling_agent(llm, tools, total_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_user_memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    response = agent_with_chat_history.stream(
        {"input": question}, config={"configurable": {"session_id": session_id}}
    )
    return response
