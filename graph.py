from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage
import operator


from load_dotenv import load_dotenv
from data.schwartz_values import schwartz_values_conflicts
import os

from collections import defaultdict
from typing import Dict, TypedDict, Annotated, Optional

import json

load_dotenv()

base_llm = ChatOpenAI(model='gpt-4')

schwartz_values = list(schwartz_values_conflicts.keys())

class GraphState(TypedDict):
    baseline_questions: Optional[Annotated[Dict[str, str], operator.add]] = None
    baseline_answers: Optional[Dict[str, bool]] = None
    value_scores: Optional[Dict[str, float]] = None
    conflicting_pairs: Optional[Dict[tuple, float]] = None 
    inconsistencies: Optional[list[Dict[tuple, str]]]
    age:int 
    profession:str


def generate_baseline_questions(state):
    fpath = "./prompts/baseline_gen_template.jinja2"
    BASELINE_TEMPLATE = PromptTemplate.from_file(fpath)
    chain = LLMChain(llm=base_llm, prompt=BASELINE_TEMPLATE)
    chain_result = chain.run(profession=state["profession"], age=state["age"])
    return {"baseline_questions": json.loads(chain_result)}


def get_baseline_answers(state):
    print("Baseline Configuration.")
    value_scores = defaultdict(int)
    assert(state.get("baseline_questions"))
    questions_dict = state.get("baseline_questions")
    for value in schwartz_values:
        questions_for_value = questions_dict[value]
        question_1 = questions_for_value.get("Question 1")
        question_2 = questions_for_value.get("Question 2")
        for question in [question_1, question_2]:
            answer = input(f">> {question}: ")
            if 'yes' in answer.lower():
                value_scores[value] += 0.5
            else:
                value_scores[value] -= 0.5
    print("Baseline Configuration Finished.")
    return {"value_scores": value_scores}
    


def get_conflicting_scenarios(state):
    fpath = "./prompts/conflicting_scenarios_template.jinja2"
    CONFLICTING_TEMPLATE = PromptTemplate.from_file(fpath)
    conflicting_pairs = state.get("conflicting_pairs") or defaultdict(float)
    inconsistencies = state.get("inconsistencies", [])
    for base_value in schwartz_values:
        conflicting_values = schwartz_values_conflicts.get(base_value)
        for c_i in conflicting_values:
            pair = (base_value, c_i)
            anti_pair = (c_i, base_value)
            chain = LLMChain(llm=base_llm, prompt=CONFLICTING_TEMPLATE)
            chain_result = json.loads(chain.run(profession=state["profession"], age=state["age"], v1=base_value, v2=c_i))
            question = chain_result["scenario"]
            answer = input(f">> {question}: ")
            if 'yes' in answer.lower(): #they prefer b_v:
                if conflicting_pairs[anti_pair] > 0:  # Conflicting preference previously recorded
                    inconsistencies.append({
                        "pair": pair,
                        "chosen": base_value,
                        "question": question
                    })
                conflicting_pairs[pair] = 1
            else: #they prefer c_i
                if(conflicting_pairs[anti_pair] <0):
                    inconsistencies.append({
                        "pair":pair,
                        "chosen":c_i,
                        "question":question
                    })                
            conflicting_pairs[pair] = -1
    return {"conflicting_pairs": conflicting_pairs}




graph = StateGraph(GraphState)
graph.add_node("baseline_generator", generate_baseline_questions)
graph.add_node("baseline_get_answers", get_baseline_answers)
graph.add_node("consolidate_inconsistencies", get_conflicting_scenarios)
graph.add_edge("baseline_generator", "baseline_get_answers")
graph.add_edge("baseline_get_answers", "consolidate_inconsistencies")
graph.add_edge("consolidate_inconsistencies", END)
graph.set_entry_point("baseline_generator")
runnable = graph.compile()


initial_state = {"baseline_questions": {}, "profession": "Architect", "age":30}


result = runnable.invoke(initial_state)

print(result)