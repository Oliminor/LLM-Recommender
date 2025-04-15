import re
import openai
from Drone.prompts import AGENT_SYSTEM_PROMPT
from Drone.tools import AGENT_TOOLS

def run_agent(user_input, model="gpt-4o-mini"):
    history = [AGENT_SYSTEM_PROMPT, {"role": "user", "content": user_input}]
    all_steps = []

    while True:
        response = openai.ChatCompletion.create(
            model=model,
            messages=history,
            temperature=0
        )
        reply = response["choices"][0]["message"]["content"]
        all_steps.append(reply)
        history.append({"role": "assistant", "content": reply})

        if "Final Answer:" in reply:
            return reply.split("Final Answer:")[-1].strip(), all_steps

        action_match = re.search(r"Action:\s*(\w+)", reply)
        input_match = re.search(r"Action Input:\s*(.*)", reply)

        if action_match and input_match:
            action = action_match.group(1)
            action_input = input_match.group(1).strip()

            if action not in AGENT_TOOLS:
                raise Exception(f"Unknown tool: {action}")

            observation = AGENT_TOOLS[action](action_input)
            history.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })
        else:
            raise Exception("Agent response missing Action and Input.")
