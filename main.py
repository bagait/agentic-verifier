import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# --- Setup ---
load_dotenv()
console = Console()

class Agent:
    """A base class for AI agents using the Groq API."""
    def __init__(self, model="llama3-8b-8192"):
        try:
            self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            self.model = model
        except Exception as e:
            console.print(f"[bold red]Error initializing Groq client: {e}[/bold red]")
            console.print("[yellow]Please make sure you have a .env file with a valid GROQ_API_KEY.[/yellow]")
            sys.exit(1)

    def run(self, system_prompt, user_prompt, response_format=None):
        """Runs the agent with given system and user prompts."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        kwargs = {
            "messages": messages,
            "model": self.model,
            "temperature": 0.2, # Lower temperature for more deterministic outputs
        }
        if response_format:
            kwargs["response_format"] = response_format
            
        try:
            chat_completion = self.client.chat.completions.create(**kwargs)
            return chat_completion.choices[0].message.content
        except Exception as e:
            console.print(f"[bold red]API Error during generation: {e}[/bold red]")
            return ""

class GenerativeAgent(Agent):
    """An agent that generates content on a given topic."""
    def __init__(self):
        super().__init__()
        self.system_prompt = (
            "You are a neutral and objective content creator. Your task is to write a concise, "
            "informative paragraph about the following topic. Focus on well-established facts and avoid speculation."
        )

    def generate(self, topic):
        """Generates a paragraph for the given topic."""
        return self.run(self.system_prompt, f"TOPIC: {topic}")

class VerificationAgent(Agent):
    """An agent that fact-checks claims against web search results."""
    def __init__(self):
        super().__init__(model="llama3-70b-8192") # Use a more powerful model for reasoning

    def _extract_claims(self, text):
        """Extracts factual claims from a body of text."""
        system_prompt = (
            "You are a claim extraction expert. Your task is to identify and list all distinct factual claims from the provided text. "
            "A factual claim is a statement that can be proven true or false. "
            "Respond ONLY with a valid JSON object containing a single key 'claims' which maps to a list of strings. "
            "For example: {\"claims\": [\"Claim 1.\", \"Claim 2.\"]}"
        )
        user_prompt = f"TEXT TO ANALYZE:\n\"\"\"\n{text}\n\"\"\""
        
        response = self.run(system_prompt, user_prompt, response_format={"type": "json_object"})
        if not response:
            return []
            
        try:
            parsed = json.loads(response)
            return parsed.get("claims", [])
        except json.JSONDecodeError:
            console.print("[bold red]Error: Failed to decode JSON from claims extraction response.[/bold red]")
            console.print(f"[yellow]Raw response was: {response}[/yellow]")
            return []

    def _search_web(self, query, max_results=3):
        """Performs a web search and returns concatenated snippets."""
        try:
            ddgs = DDGS() # Instantiate locally to be thread-safe
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return ""
            return "\n".join([r.get('body', '') for r in results])
        except Exception as e:
            console.print(f"[bold red]Error during web search for '{query}': {e}[/bold red]")
            return ""

    def _verify_claim(self, claim, search_results):
        """Verifies a single claim against search results."""
        if not search_results:
            return "UNVERIFIABLE"
        
        system_prompt = (
            "You are a meticulous fact-checker. Your task is to verify a given claim using only the provided search results as evidence. "
            "Analyze the search results and determine if they support, contradict, or fail to provide enough information to verify the claim. "
            "Do not use your own knowledge. Base your conclusion solely on the text provided.\n\n"
            "Respond with exactly one word: VERIFIED, CONTRADICTED, or UNVERIFIABLE."
        )
        user_prompt = (
            f"CLAIM TO VERIFY: \"{claim}\"\n\n"
            f"SEARCH RESULTS:\n\"\"\"\n{search_results}\n\"\"\"\n\n"
            "Based *only* on the search results, what is the verification status?"
        )
        
        response = self.run(system_prompt, user_prompt)
        if not response:
            return "UNVERIFIABLE"
            
        response = response.strip().upper()
        
        if "UNVERIFIABLE" in response:
            return "UNVERIFIABLE"
        elif "CONTRADICTED" in response:
            return "CONTRADICTED"
        elif "VERIFIED" in response:
            return "VERIFIED"
            
        return "UNVERIFIABLE"

    def verify(self, text):
        """Performs the full verification process on a piece of text."""
        console.print("[bold cyan]Step 1: Extracting factual claims...[/bold cyan]")
        claims = self._extract_claims(text)
        if not claims:
            console.print("[yellow]No factual claims were extracted.[/yellow]")
            return []
        
        console.print(f"[green]Found {len(claims)} claims to verify.[/green]")
        
        verification_results = []
        
        def process_claim(claim):
            search_results = self._search_web(claim)
            status_result = self._verify_claim(claim, search_results)
            return {"claim": claim, "status": status_result}

        with console.status(f"[bold cyan]Step 2: Verifying claims (0/{len(claims)} completed)...[/bold cyan]") as status:
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_claim = {executor.submit(process_claim, claim): claim for claim in claims}
                for i, future in enumerate(as_completed(future_to_claim)):
                    result = future.result()
                    verification_results.append(result)
                    status.update(f"[bold cyan]Step 2: Verifying claims ({i+1}/{len(claims)} completed)...[/bold cyan]")

        # Restore original order since threads complete out of order
        claim_to_result = {res["claim"]: res for res in verification_results}
        ordered_results = [claim_to_result[claim] for claim in claims]

        return ordered_results

def main():
    """Main function to run the agentic verifier system."""
    parser = argparse.ArgumentParser(description="Agentic Verifier System")
    parser.add_argument("--topic", type=str, help="Topic for the generative agent to write about")
    args = parser.parse_args()

    console.print(Panel.fit("[bold blue]Agentic Verifier System[/bold blue]", 
                              subtitle="[cyan]Generate & Fact-Check AI Content[/cyan]"))

    topic = args.topic
    if not topic:
        topic = console.input("\nEnter a topic for the generative agent to write about: ")

    # Generation Step
    with console.status("[bold cyan]Generating initial content...[/bold cyan]"):
        generative_agent = GenerativeAgent()
        generated_text = generative_agent.generate(topic)
    
    if not generated_text:
        console.print("[bold red]Failed to generate content. Exiting.[/bold red]")
        sys.exit(1)
        
    console.print(Panel(Markdown(generated_text), title="[bold green]Generated Content[/bold green]", border_style="green"))

    # Verification Step
    console.print(Panel.fit("[bold blue]Starting Verification Process[/bold blue]"))
    verification_agent = VerificationAgent()
    results = verification_agent.verify(generated_text)

    # Display Results
    if results:
        console.print(Panel.fit("[bold blue]Verification Report[/bold blue]"))
        report_markdown = """| Status | Claim |\n|:---:|:---|\n"""
        status_emojis = {
            "VERIFIED": "\u2705",
            "CONTRADICTED": "\u274c",
            "UNVERIFIABLE": "\u2753"
        }
        for result in results:
            emoji = status_emojis.get(result['status'], '\u2753')
            report_markdown += f"| {emoji} {result['status']} | {result['claim']} |\n"
        
        console.print(Markdown(report_markdown))
    else:
        console.print("[yellow]Verification process complete. No results to show.[/yellow]")

if __name__ == "__main__":
    main()
