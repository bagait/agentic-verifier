# Agentic Verifier: AI Fact-Checking AI

This project demonstrates a simple but powerful multi-agent system where one AI agent generates content and a second, independent 'verifier' agent fact-checks the generated claims using real-time web search. This pattern is crucial for building more reliable and trustworthy AI systems that can ground their outputs in verifiable facts.

Inspired by the growing concern over AI-generated misinformation, this tool provides a practical implementation of an automated verification layer.

## Features

- **Generative Agent**: Creates a concise, factual paragraph on a user-provided topic.
- **Verification Agent**: A sophisticated agent that:
    1.  **Extracts Claims**: Intelligently identifies distinct, factual claims using forced JSON mode for high reliability.
    2.  **Parallel Web Search**: Uses DuckDuckGo to search for evidence related to each claim concurrently, drastically speeding up verification.
    3.  **Analyzes Evidence**: A powerful LLM (`llama3-70b-8192`) analyzes the search results to determine if a claim is `VERIFIED`, `CONTRADICTED`, or `UNVERIFIABLE`.
- **Rich Console Output**: Presents the entire process and the final report in a clean, easy-to-read format.
- **CLI Support**: Pass topics directly via the command line.

## How It Works

The system follows a two-step process:

1.  **Generation**: The `GenerativeAgent` is given a topic and prompted to write a neutral, factual summary. It uses a smaller, faster model (`llama3-8b-8192`) for this task.

2.  **Verification**: The `VerificationAgent` takes the output from the first agent and begins its process:
    - It first sends the text to an LLM with a specific prompt designed to extract all factual claims and return them as a JSON object.
    - For each extracted claim, it performs a web search using the `duckduckgo-search` library concurrently.
    - It then provides a powerful LLM (`llama3-70b-8192`) with the claim and the search result snippets. The LLM's only job is to compare the two and output a single status word: `VERIFIED`, `CONTRADICTED`, or `UNVERIFIABLE`.
    - Finally, all results are compiled into a summary report for the user.

This separation of concerns—generation vs. verification—and the use of external, real-time data from a search engine are key principles for mitigating LLM hallucinations and building more robust AI applications.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bagait/agentic-verifier.git
    cd agentic-verifier
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set up your API Key:**
    - You will need a free Groq API key from [GroqCloud](https://console.groq.com/keys).
    - Create a file named `.env` in the project root.
    - Add your API key to the `.env` file:
      ```env
      GROQ_API_KEY="gsk_YourSecretKeyGoesHere"
      ```

## Usage

Run the `main.py` script from your terminal. You can either be prompted for a topic or pass one directly:

```bash
python main.py --topic "The discovery of penicillin"
```

The script will take over, and you will see the generation and verification process live in your terminal.

### Example

```
$ python main.py --topic "The discovery of penicillin"

... (output from the tool) ...

┌────────────────────────────────────────────────────────── Verification Report ──────────────────────────────────────────────────────────┐
│ | Status      | Claim                                                                                                                   |
│ |:-----------:|:------------------------------------------------------------------------------------------------------------------------|
│ | ✅ VERIFIED   | Penicillin was discovered by Scottish physician and microbiologist Alexander Fleming in 1928.                           |
│ | ✅ VERIFIED   | The discovery happened by accident when Fleming noticed that a mold, Penicillium notatum, had contaminated a petri dish.|
│ | ✅ VERIFIED   | This mold was preventing the growth of Staphylococcus bacteria.                                                         |
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
