import os
import json
import subprocess
import glob
import re
import operator
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List, TypedDict, Annotated, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from moviepy import VideoFileClip, concatenate_videoclips

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- CONFIGURATION ---
LLM_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3


class ScenePlan(BaseModel):
    id: int = Field(description="Scene number, starting from 1")
    duration: str = Field(description="Duration string (e.g. '10s')")
    visual_description: str = Field(description="Visual instructions for Manim")
    search_keywords: str = Field(
        description="Keywords to search docs (e.g. 'Manim NeuralNetworkMobject', 'Manim 3D Camera')")


class VideoPlan(BaseModel):
    scenes: List[ScenePlan]


class SceneCode(TypedDict):
    id: int
    code: str


class AgentState(TypedDict):
    topic: str
    plan: List[ScenePlan]
    codes: List[SceneCode]

    # Execution State
    current_index: int
    current_context: str  # Stores search results for the coder
    current_retry_count: int
    current_error: Optional[str]
    current_code: str
    video_paths: Annotated[List[str], operator.add]


# --- NODES ---

def planner_node(state: AgentState):
    """Generates the plan including search keywords."""
    print(f"\n[Phase 1] Planning topic: '{state['topic']}'...")

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)
    parser = PydanticOutputParser(pydantic_object=VideoPlan)

    prompt = ChatPromptTemplate.from_template(
        """
        You are a Manim Video Architect.
        1. Break the topic into a storyboard.
        2. Total video length is 50 seconds not more .
        3. Break it into scenes where each scene is STRICTLY under 12 seconds.
        4. For each scene, provide specific SEARCH KEYWORDS to find Manim documentation (e.g., 'Manim draw graph edges', 'Manim FadeIn animation').
        5. Use simple Visuals and combine them 
        
        {format_instructions}

        Topic: {topic}
        """
    )

    chain = prompt | llm | parser

    try:
        plan = chain.invoke({
            "topic": state['topic'],
            "format_instructions": parser.get_format_instructions()
        })

        with open("scenes.json", "w") as f:
            json.dump([s.dict() for s in plan.scenes], f, indent=2)
        print(f"--> Saved plan ({len(plan.scenes)} scenes).")

        return {"plan": plan.scenes, "current_index": 0, "codes": []}

    except Exception as e:
        print(f"Planning failed: {e}")
        raise e


def researcher_node(state: AgentState):
    """Searches Manim docs for the current scene."""
    idx = state['current_index']
    scene_data = state['plan'][idx]
    keywords = scene_data.search_keywords

    print(f"[Phase 2 - Step A] Searching docs for Scene {scene_data.id}: '{keywords}'...")

    search = DuckDuckGoSearchRun()
    try:
        # Search for Manim specific documentation
        # We append "Manim python library" to ensure technical results
        query = f"Manim Community Edition python {keywords} code example"
        results = search.invoke(query)
        print(f"   --> Found docs: {results[:100]}...")  # Print preview
        return {"current_context": results}
    except Exception as e:
        print(f"   --> Search failed: {e}. Proceeding without docs.")
        return {"current_context": "No documentation found."}


def coder_node(state: AgentState):
    """Generates code using the Plan + Search Results."""
    idx = state['current_index']
    scene_data = state['plan'][idx]
    docs = state['current_context']

    print(f"[Phase 2 - Step B] Coding Scene {scene_data.id}...")

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1)
    prompt = ChatPromptTemplate.from_template(
        """
        You are a Manim Expert. Write Python code for Scene {id}.

        VISUAL PLAN: {visual}
        DURATION: {duration}

        RELEVANT DOCUMENTATION/EXAMPLES FOUND:
        {docs}

        STRICT RULES:
        1. Class name MUST be 'Scene{id}'.
        2. Inherit from 'Scene'.
        3. Use 'from manim import *'.
        4. Use the documentation provided above to use correct classes/methods.
        5. If the documentation shows complex Mobjects, simplify if needed to avoid errors.
        6. NO markdown. Output raw code only.
        """
    )

    chain = prompt | llm
    res = chain.invoke({
        "id": scene_data.id,
        "visual": scene_data.visual_description,
        "duration": scene_data.duration,
        "docs": docs
    })

    raw_code = res.content.replace("```python", "").replace("```", "").strip()

    current_codes = list(state.get('codes', []))
    current_codes.append({"id": scene_data.id, "code": raw_code})

    return {"codes": current_codes}


def saver_node(state: AgentState):
    """Saves codes to JSON."""
    print("\n[Phase 2 Complete] Saving all codes...")
    sorted_codes = sorted(state['codes'], key=lambda x: x['id'])
    with open("codes.json", "w") as f:
        json.dump(sorted_codes, f, indent=2)
    return {"current_index": 0}


def renderer_node(state: AgentState):
    """Renders the scene."""
    idx = state['current_index']
    if idx >= len(state['plan']): return {"current_error": "Index out of bounds"}

    scene_id = state['plan'][idx].id
    code_entry = next((c for c in state['codes'] if c['id'] == scene_id), None)

    if not code_entry: return {"current_error": "No code found"}

    print(f"[Phase 3] Rendering Scene {scene_id}...")
    file_name, class_name = f"scene_{scene_id}.py", f"Scene{scene_id}"

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(code_entry['code'])

    cmd = ["manim", "-qh", "--media_dir", "./media", file_name, class_name]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            error_log = result.stderr[-1000:]
            print(f"!!! Render Error:\n{error_log}")
            return {"current_error": error_log, "current_code": code_entry['code']}

        search_path = f"media/videos/{file_name.replace('.py', '')}"
        found_videos = glob.glob(f"{search_path}/**/*.mp4", recursive=True)

        if found_videos:
            found_videos.sort(key=os.path.getmtime, reverse=True)
            return {"video_paths": [found_videos[0]], "current_error": None}
        return {"current_error": "File not created", "current_code": code_entry['code']}

    except Exception as e:
        return {"current_error": str(e), "current_code": code_entry['code']}


def debugger_node(state: AgentState):
    """Fixes code based on render error."""
    print(f"  [!] Debugging Scene...")
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template("Fix Manim Code.\nError: {error}\nCode: {code}\nOutput ONLY code.")
    chain = prompt | llm
    res = chain.invoke({"error": state['current_error'], "code": state['current_code']})
    fixed_code = res.content.replace("```python", "").replace("```", "").strip()

    # Update code list
    scene_id = state['plan'][state['current_index']].id
    new_codes = [{"id": c['id'], "code": fixed_code if c['id'] == scene_id else c['code']} for c in state['codes']]

    return {"codes": new_codes, "current_retry_count": state['current_retry_count'] + 1, "current_error": None}


def concatenator_node(state: AgentState):
    """Stitches final video."""
    print("\n[Phase 4] Stitching videos...")
    safe_topic = re.sub(r'[^a-zA-Z0-9]', '_', state['topic'])
    output_filename = f"{safe_topic}_final.mp4"

    clips = [VideoFileClip(p) for p in state['video_paths'] if os.path.exists(p)]
    if clips:
        final = concatenate_videoclips(clips)
        final.write_videofile(output_filename, logger=None)
        print(f"SUCCESS: {os.path.abspath(output_filename)}")
        return {"video_paths": [output_filename]}
    return state


# --- ROUTING ---
def route_coding(state: AgentState):
    if state['current_index'] + 1 < len(state['plan']): return "next_research"
    return "save_codes"


def route_rendering(state: AgentState):
    if state['current_error']:
        return "debugger" if state['current_retry_count'] < MAX_RETRIES else "next_scene"
    return "next_scene"


def route_next_render(state: AgentState):
    if state['current_index'] + 1 <= len(state['plan']): return "render_loop"
    return "finalize"


def increment_index(state: AgentState):
    return {"current_index": state['current_index'] + 1, "current_retry_count": 0, "current_error": None}


# --- GRAPH BUILDER ---
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)  # NEW NODE
    workflow.add_node("coder", coder_node)
    workflow.add_node("saver", saver_node)
    workflow.add_node("renderer", renderer_node)
    workflow.add_node("debugger", debugger_node)
    workflow.add_node("concatenator", concatenator_node)
    workflow.add_node("incrementor_code", increment_index)
    workflow.add_node("incrementor_render", increment_index)

    workflow.set_entry_point("planner")

    # Research -> Code -> Loop
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "coder")

    workflow.add_conditional_edges("coder", route_coding, {
        "next_research": "incrementor_code",
        "save_codes": "saver"
    })
    workflow.add_edge("incrementor_code", "researcher")  # Loop back to research

    # Rendering
    workflow.add_edge("saver", "renderer")
    workflow.add_conditional_edges("renderer", route_rendering, {
        "debugger": "debugger",
        "next_scene": "incrementor_render"
    })
    workflow.add_edge("debugger", "renderer")
    workflow.add_conditional_edges("incrementor_render", route_next_render, {
        "render_loop": "renderer",
        "finalize": "concatenator"
    })
    workflow.add_edge("concatenator", END)

    return workflow.compile()


topic = input("Enter Topic: ")

initial_state = {
    "topic": topic,
    "plan": [],
    "codes": [],
    "current_index": 0,
    "current_retry_count": 0,
    "current_error": None,
    "current_code": "",
    "current_context": "",
    "video_paths": []
}

app = build_graph()

print("Starting Agent (Limit increased to 150 steps)...")

# --- FIX HERE: Add config with recursion_limit ---
app.invoke(
    initial_state,
    config={"recursion_limit": 150}
)

