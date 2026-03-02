import os
import re
import json
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
            exit()

    def run(self, prompt):
        """Runs the agent with a given prompt."""
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
        )
        return chat_completion.choices[0].message.content

class GenerativeAgent(Agent):
    """An agent that generates content on a given topic."""
    def __init__(self):
        super().__init__()
        self.prompt_template = (
            "You are a neutral and objective content creator. Your task is to write a concise, "
            "informative paragraph about the following topic. Focus on well-established facts and avoid speculation."
            "\n\nTOPIC: {topic}"
        )

    def generate(self, topic):
        """Generates a paragraph for the given topic."""
        prompt = self.prompt_template.format(topic=topic)
        return self.run(prompt)

class VerificationAgent(Agent):
    """An agent that fact-checks claims against web search results."""
    def __init__(self):
        super().__init__(model="llama3-70b-8192") # Use a more powerful model for reasoning
        self.ddgs = DDGS()

    def _extract_claims(self, text):
        """Extracts factual claims from a body of text."""
        prompt = (
            "You are a claim extraction expert. Your task is to identify and list all distinct factual claims from the provided text. "
            "A factual claim is a statement that can be proven true or false. Present these claims as a JSON list of strings."
            "\n\nTEXT TO ANALYZE:\n\"\"\"\n{text}\n\"\"\"\n\nRespond ONLY with a valid JSON list of strings. For example: [\"Claim 1.\", \"Claim 2.\"]"
        ).format(text=text)
        
        response = self.run(prompt)
        try:
            # Use regex to find the JSON list within potential markdown fences
            match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if match:
                json_str = match.group(1)
                claims = json.loads(f"[{json_str}]")
                return claims
            else:
                # Fallback for plain JSON output
                return json.loads(response)
        except json.JSONDecodeError:
            console.print(f"[bold red]Error: Failed to decode JSON from claims extraction response.[/bold red]")
            console.print(f"[yellow]Raw response was: {response}[/yellow]")
            return []

    def _search_web(self, query, max_results=3):
        """Performs a web search and returns concatenated snippets."""
        try:
            results = list(self.ddgs.text(query, max_results=max_results))
            return "\n".join([r['body'] for r in results])
        except Exception as e:
            console.print(f"[bold red]Error during web search for '{query}': {e}[/bold red]")
            return ""

    def _verify_claim(self, claim, search_results):
        """Verifies a single claim against search results."""
        if not search_results:
            return "UNVERIFIABLE"
        
        prompt = (
            "You are a meticulous fact-checker. Your task is to verify a given claim using only the provided search results as evidence. "
            "Analyze the search results and determine if they support, contradict, or fail to provide enough information to verify the claim. "
            "Do not use your own knowledge. Base your conclusion solely on the text provided."
            "\n\nCLAIM TO VERIFY: \"{claim}\""
            "\n\nSEARCH RESULTS:\n\"\"\"\n{search_results}\n\"\"\"\n\n"
            "Based *only* on the search results, what is the verification status? "
            "Respond with a single word: VERIFIED, CONTRADICTED, or UNVERIFIABLE."
        ).format(claim=claim, search_results=search_results)
        
        response = self.run(prompt).strip().upper()
        if response not in ["VERIFIED", "CONTRADICTED", "UNVERIFIABLE"]:
            return "UNVERIFIABLE" # Default to unverifiable if the model misbehaves
        return response

    def verify(self, text):
        """Performs the full verification process on a piece of text."""
        console.print("[bold cyan]Step 1: Extracting factual claims...[/bold cyan]")
        claims = self._extract_claims(text)
        if not claims:
            console.print("[yellow]No factual claims were extracted.[/yellow]")
            return []
        
        console.print(f"[green]Found {len(claims)} claims to verify.[/green]")
        
        verification_results = []
        for i, claim in enumerate(claims):
            with console.status(f"[bold cyan]Step 2: Verifying claim {i+1}/{len(claims)}: '{claim}'...[/bold cyan]") as status:
                status.update("[magenta]Searching the web...[/magenta]")
                search_results = self._search_web(claim)
                
                status.update("[magenta]Analyzing search results...[/magenta]")
                status_result = self._verify_claim(claim, search_results)
                verification_results.append({"claim": claim, "status": status_result})

        return verification_results

def main():
    """Main function to run the agentic verifier system."""
    console.print(Panel.fit("[bold blue]Agentic Verifier System[/bold blue]", 
                              subtitle="[cyan]Generate & Fact-Check AI Content[/cyan]"))

    topic = console.input("\nEnter a topic for the generative agent to write about: ")

    # Generation Step
    with console.status("[bold cyan]Generating initial content...[/bold cyan]"):
        generative_agent = GenerativeAgent()
        generated_text = generative_agent.generate(topic)
    
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
            "VERIFIED": "✅",
            "CONTRADICTED": "❌",
            "UNVERIFIABLE": "❓"
        }
        for result in results:
            emoji = status_emojis.get(result['status'], '❓')
            report_markdown += f"| {emoji} {result['status']} | {result['claim']} |\n"
        
        console.print(Markdown(report_markdown))
    else:
        console.print("[yellow]Verification process complete. No results to show.[/yellow]")

if __name__ == "__main__":
    main()
