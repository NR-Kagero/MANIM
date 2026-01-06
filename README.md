AGENTIC MANIM VIDEO GENERATOR


An AI-powered automation tool that takes a user-provided topic, plans a 
storyboard, researches necessary Manim documentation, generates Python code, 
renders animations, self-corrects errors, and stitches the final video together.

Built with: LangGraph, LangChain, Google Gemini, and Manim.

------------------------------------------------------------------------------
FEATURES
------------------------------------------------------------------------------
* **Automated Planning**: Breaks down complex topics into a 50-second storyboard 
    divided into logical scenes.
* **Active Research**: Uses DuckDuckGo to search for specific Manim documentation
    and code examples relevant to the visual plan.
* **Code Generation**: Writes executable Manim Python code for each scene.
* **Self-Healing (Debugging)**: If a scene fails to render (syntax error, 
    invalid method), the agent analyzes the error log and rewrites the code 
    automatically (up to 3 retries per scene).
* **Video Stitching**: Automatically concatenates all rendered scenes into a 
    single final MP4 file.

------------------------------------------------------------------------------
PREREQUISITES
------------------------------------------------------------------------------
1.  **Python 3.10+**
2.  **System Dependencies for Manim**:
    Manim requires FFmpeg, OpenGL, and LaTeX to be installed on your system.
    * Windows: Install FFmpeg and MiKTeX.
    * MacOS: `brew install ffmpeg mactex cairo pango`
    * Linux: `sudo apt install ffmpeg texlive-full libcairo2-dev libpango1.0-dev`
    *(See https://docs.manim.community/en/stable/installation.html for details)*
3.  **Google Gemini API Key**: You need a valid API key from Google AI Studio.

------------------------------------------------------------------------------
INSTALLATION
------------------------------------------------------------------------------
1.  Clone or download this repository.

2.  Install the required Python libraries via pip:

    pip install langgraph langchain-google-genai langchain-community \
    duckduckgo-search manim moviepy python-dotenv pydantic

------------------------------------------------------------------------------
CONFIGURATION
------------------------------------------------------------------------------
1.  Create a file named `.env` in the root directory of the project.
2.  Add your Google API key to the file:

    GOOGLE_API_KEY=your_actual_api_key_here

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
1.  Run the script:

    python AgenticManim.py

2.  When prompted, enter the topic you want the video to be about.
    * Example: "Pythagorean Theorem"
    * Example: "How Neural Networks learn"
    * Example: "The area of a circle derivation"

3.  The agent will perform the following steps (visible in the console):
    * [Phase 1] Plan the scenes.
    * [Phase 2] Research Manim docs and generate code for all scenes.
    * [Phase 3] Render scenes one by one (including auto-debugging).
    * [Phase 4] Stitch the videos.

4.  **Output**:
    * Intermediate scene codes are saved in `codes.json`.
    * The final video is saved as `[Topic_Name]_final.mp4`.

------------------------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------------------------
AgenticManim.py  # Main script containing the LangGraph workflow
.env             # API Keys (user created)
media/           # Folder created by Manim containing raw renders
scenes.json      # Temporary file storing the generated storyboard
codes.json       # Temporary file storing the generated Python code

------------------------------------------------------------------------------
HOW IT WORKS (THE GRAPH)
------------------------------------------------------------------------------
The system operates as a state machine using LangGraph:

1.  **Planner**: Uses Gemini to create a JSON plan containing visual descriptions
    and specific search keywords.
2.  **Researcher**: Searches the web for "Manim [keyword] code example" to
    get up-to-date syntax.
3.  **Coder**: Writes the `Scene` class inheriting from Manim.
4.  **Renderer**: Executes the code via subprocess.
5.  **Debugger**: (Conditional) If Renderer fails, this node reads the 
    stderr output and asks the LLM to fix the code.
6.  **Concatenator**: Uses MoviePy to combine successful renders.

------------------------------------------------------------------------------
TROUBLESHOOTING
------------------------------------------------------------------------------
* **LaTeX Errors**: If Manim fails to render text, ensure you have a standard
    LaTeX distribution (MiKTeX or TeX Live) installed and added to your PATH.
* **Timeout**: Complex scenes may time out (set to 180s in the script). 
    You can increase the timeout in the `renderer_node` function.
* **Google API Limit**: If you hit quota limits, wait a minute and try again, 
    or ensure you are using a paid tier/valid key.
