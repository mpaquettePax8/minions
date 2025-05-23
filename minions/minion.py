from typing import List, Dict, Any
import json
import re
import os
from datetime import datetime

from minions.clients import OpenAIClient, TogetherClient, GeminiClient

from minions.prompts.minion import (
    SUPERVISOR_CONVERSATION_PROMPT,
    SUPERVISOR_FINAL_PROMPT,
    SUPERVISOR_INITIAL_PROMPT,
    WORKER_SYSTEM_PROMPT,
    REMOTE_SYNTHESIS_COT,
    REMOTE_SYNTHESIS_FINAL,
    WORKER_PRIVACY_SHIELD_PROMPT,
    REFORMAT_QUERY_PROMPT,
)

from minions.usage import Usage

# Override the supervisor initial prompt to encourage task decomposition.
SUPERVISOR_INITIAL_PROMPT = """\
We need to perform the following task.

### Task
{task}

### Instructions
You will not have direct access to the context, but you can chat with a small language model that has read the entire content.

Let's use an incremental, step-by-step approach to ensure we fully decompose the task before proceeding. Please follow these steps:

1. Decompose the Task:
   Break down the overall task into its key components or sub-tasks. Identify what needs to be done and list these sub-tasks.

2. Explain Each Component:
   For each sub-task, briefly explain why it is important and what you expect it to achieve. This helps clarify the reasoning behind your breakdown.

3. Formulate a Focused Message:
   Based on your breakdown, craft a single, clear message to send to the small language model. This message should represent one focused sub-task derived from your decomposition.

4. Conclude with a Final Answer:  
   After your reasoning, please provide a **concise final answer** that directly and conclusively addresses the original task. Make sure this final answer includes all the specific details requested in the task.

Your output should be in the following JSON format:

```json
{{
    "reasoning": "<your detailed, step-by-step breakdown here>",
    "message": "<your final, focused message to the small language model>"
}}
"""

# Override the final response prompt to encourage a more informative final answer
REMOTE_SYNTHESIS_FINAL = """\
Here is the detailed response from the step-by-step reasoning phase.

### Detailed Response
{response}

### Instructions
Based on the detailed reasoning above, synthesize a clear and informative final answer that directly addresses the task with all the specific details required. In your final answer, please:

1. Summarize the key findings and reasoning steps.
2. Clearly state the conclusive answer, incorporating the important details.
3. Ensure the final answer is self-contained and actionable.

If you determine that you have gathered enough information to fully answer the task, output the following JSON with your final answer:

```json
{{
    "decision": "provide_final_answer", 
    "answer": "<your detailed, conclusive final answer here>"
}}
```

Otherwise, if the task is not complete, request the small language model to do additional work, by outputting the following:

```json
{{
    "decision": "request_additional_info",
    "message": "<your message to the small language model>"
}}
```

"""

def _escape_newlines_in_strings(json_str: str) -> str:
    # This regex naively matches any content inside double quotes (including escaped quotes)
    # and replaces any literal newline characters within those quotes.
    # was especially useful for anthropic client
    return re.sub(
        r'(".*?")',
        lambda m: m.group(1).replace("\n", "\\n"),
        json_str,
        flags=re.DOTALL,
    )


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from text that may be wrapped in markdown code blocks."""
    block_matches = list(re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL))
    bracket_matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))

    if block_matches:
        json_str = block_matches[-1].group(1).strip()
    elif bracket_matches:
        json_str = bracket_matches[-1].group(0)
    else:
        json_str = text

    # Minimal fix: escape newlines only within quoted JSON strings.
    json_str = _escape_newlines_in_strings(json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {json_str}")
        raise


class Minion:
    def __init__(
        self,
        local_client=None,
        remote_client=None,
        max_rounds=3,
        callback=None,
        log_dir="minion_logs",
    ):
        """Initialize the Minion with local and remote LLM clients.

        Args:
            local_client: Client for the local model (e.g. OllamaClient)
            remote_client: Client for the remote model (e.g. OpenAIClient)
            max_rounds: Maximum number of conversation rounds
            callback: Optional callback function to receive message updates
        """
        self.local_client = local_client
        self.remote_client = remote_client
        self.max_rounds = max_rounds
        self.callback = callback
        self.log_dir = log_dir
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

    def __call__(
        self,
        task: str,
        context: List[str],
        max_rounds=None,
        doc_metadata=None,
        logging_id=None,  # this is the name/id to give to the logging .json file
        is_privacy=False,
        images=None,
    ):
        """Run the minion protocol to answer a task using local and remote models.

        Args:
            task: The task/question to answer
            context: List of context strings
            max_rounds: Override default max_rounds if provided
            doc_metadata: Optional metadata about the documents
            logging_id: Optional identifier for the task, used for named log files

        Returns:
            Dict containing final_answer, conversation histories, and usage statistics
        """

        print("\n========== MINION TASK STARTED ==========")
        print(f"Task: {task}")
        print(f"Max rounds: {max_rounds or self.max_rounds}")
        print(f"Privacy enabled: {is_privacy}")
        print(f"Images provided: {True if images else False}")

        if max_rounds is None:
            max_rounds = self.max_rounds

        # Join context sections
        context = "\n\n".join(context)
        print(f"Context length: {len(context)} characters")

        # Initialize the log structure
        conversation_log = {
            "task": task,
            "context": context,
            "conversation": [],
            "generated_final_answer": "",
            "usage": {
                "remote": {},
                "local": {},
            },
        }

        # Initialize message histories and usage tracking
        supervisor_messages = [
            {
                "role": "user",
                "content": SUPERVISOR_INITIAL_PROMPT.format(
                    task=task, max_rounds=max_rounds
                ),
            }
        ]

        # Add initial supervisor prompt to conversation log
        conversation_log["conversation"].append(
            {
                "user": "remote",
                "prompt": SUPERVISOR_INITIAL_PROMPT.format(
                    task=task, max_rounds=max_rounds
                ),
                "output": None,
            }
        )

        # print whether privacy is enabled
        print("Privacy is enabled: ", is_privacy)

        remote_usage = Usage()
        local_usage = Usage()

        worker_messages = []
        supervisor_messages = []

        # if privacy import from minions.utils.pii_extraction
        if is_privacy:
            from minions.utils.pii_extraction import PIIExtractor

            # Extract PII from context
            pii_extractor = PIIExtractor()
            str_context = "\n\n".join(context)
            pii_extracted = pii_extractor.extract_pii(str_context)

            # Extract PII from query
            query_pii_extracted = pii_extractor.extract_pii(task)
            reformat_query_task = REFORMAT_QUERY_PROMPT.format(
                query=task, pii_extracted=str(query_pii_extracted)
            )

            # Clean PII from query
            reformatted_task, usage, done_reason = self.local_client.chat(
                messages=[{"role": "user", "content": reformat_query_task}]
            )
            local_usage += usage
            pii_reformatted_task = reformatted_task[0]

            # Log the reformatted task
            output = f"""**PII Reformated Task:**
            {pii_reformatted_task}
            """

            if self.callback:
                self.callback("worker", output)

            # Initialize message histories
            supervisor_messages = [
                {
                    "role": "user",
                    "content": SUPERVISOR_INITIAL_PROMPT.format(
                        task=pii_reformatted_task,
                        max_rounds=max_rounds,
                    ),
                }
            ]
            worker_messages = [
                {
                    "role": "system",
                    "content": WORKER_SYSTEM_PROMPT.format(context=context, task=task),
                }
            ]
        else:
            supervisor_messages = [
                {
                    "role": "user",
                    "content": SUPERVISOR_INITIAL_PROMPT.format(
                        task=task, max_rounds=max_rounds
                    ),
                }
            ]
            worker_messages = [
                {
                    "role": "system",
                    "content": WORKER_SYSTEM_PROMPT.format(context=context, task=task),
                    "images": images,
                }
            ]

        if max_rounds is None:
            max_rounds = self.max_rounds

        # Initial supervisor call to get first question
        if self.callback:
            self.callback("supervisor", None, is_final=False)

        if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
            supervisor_response, supervisor_usage = self.remote_client.chat(
                messages=supervisor_messages, response_format={"type": "json_object"}
            )
        elif isinstance(self.remote_client, GeminiClient):
            from pydantic import BaseModel

            class output(BaseModel):
                decision: str
                message: str
                answer: str

            # how to make message and answer optional

            supervisor_response, supervisor_usage = self.remote_client.chat(
                messages=supervisor_messages,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": output,
                },
            )
        else:
            supervisor_response, supervisor_usage = self.remote_client.chat(
                messages=supervisor_messages
            )

        remote_usage += supervisor_usage
        supervisor_messages.append(
            {"role": "assistant", "content": supervisor_response[0]}
        )

        # Update the last conversation entry with the ouput
        conversation_log["conversation"][-1]["output"] = supervisor_response[0]

        if self.callback:
            self.callback("supervisor", supervisor_messages[-1])

        # Extract first question for worker
        if isinstance(self.remote_client, (OpenAIClient, TogetherClient, GeminiClient)):
            try:
                supervisor_json = json.loads(supervisor_response[0])

            except:
                try:
                    supervisor_json = _extract_json(supervisor_response[0])
                except:
                    supervisor_json = supervisor_response[0]
        else:
            supervisor_json = _extract_json(supervisor_response[0])

        worker_messages.append({"role": "user", "content": supervisor_json["message"]})

        # Add worker prompt to conversation log
        conversation_log["conversation"].append(
            {"user": "local", "prompt": supervisor_json["message"], "output": None}
        )

        final_answer = None
        for round in range(max_rounds):
            # Get worker's response
            if self.callback:
                self.callback("worker", None, is_final=False)

            worker_response, worker_usage, done_reason = self.local_client.chat(
                messages=worker_messages
            )

            print(f"Worker response: {worker_response}")
            print(f"Worker usage: {worker_usage}")

            local_usage += worker_usage

            if is_privacy:
                if self.callback:
                    output = f"""**_My output (pre-privacy shield):_**

                    {worker_response[0]}
                    """
                    self.callback("worker", output)

                worker_privacy_shield_prompt = WORKER_PRIVACY_SHIELD_PROMPT.format(
                    output=worker_response[0],
                    pii_extracted=str(pii_extracted),
                )
                worker_response, worker_usage, done_reason = self.local_client.chat(
                    messages=[{"role": "user", "content": worker_privacy_shield_prompt}]
                )
                local_usage += worker_usage

                worker_messages.append(
                    {"role": "assistant", "content": worker_response[0]}
                )
                # Update the last conversation entry with the output
                conversation_log["conversation"][-1]["output"] = worker_response[0]

                if self.callback:
                    output = f"""**_My output (post-privacy shield):_**

                    {worker_response[0]}
                    """
                    self.callback("worker", output)
            else:
                worker_messages.append(
                    {"role": "assistant", "content": worker_response[0]}
                )

                # Update the last conversation entry with the output
                conversation_log["conversation"][-1]["output"] = worker_response[0]

                if self.callback:
                    self.callback("worker", worker_messages[-1])

            # Format prompt based on whether this is the final round
            if round == max_rounds - 1:
                supervisor_prompt = SUPERVISOR_FINAL_PROMPT.format(
                    response=worker_response[0]
                )

                # Add supervisor final prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": supervisor_prompt, "output": None}
                )
            else:
                # First step: Think through the synthesis
                cot_prompt = REMOTE_SYNTHESIS_COT.format(response=worker_response[0])

                # Add supervisor COT prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": cot_prompt, "output": None}
                )

                supervisor_messages.append({"role": "user", "content": cot_prompt})

                step_by_step_response, usage = self.remote_client.chat(
                    supervisor_messages
                )

                remote_usage += usage

                supervisor_messages.append(
                    {"role": "assistant", "content": step_by_step_response[0]}
                )

                # Update the last conversation entry with the output
                conversation_log["conversation"][-1]["output"] = step_by_step_response[
                    0
                ]

                # Second step: Get structured output
                supervisor_prompt = REMOTE_SYNTHESIS_FINAL.format(
                    response=step_by_step_response[0]
                )

                # Add supervisor synthesis prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": supervisor_prompt, "output": None}
                )

            supervisor_messages.append({"role": "user", "content": supervisor_prompt})

            if self.callback:
                self.callback("supervisor", None, is_final=False)

            # Get supervisor's response
            if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
                supervisor_response, supervisor_usage = self.remote_client.chat(
                    messages=supervisor_messages,
                    response_format={"type": "json_object"},
                )
            else:
                from pydantic import BaseModel

                class remote_output(BaseModel):
                    decision: str
                    message: str
                    answer: str

                supervisor_response, supervisor_usage = self.remote_client.chat(
                    messages=supervisor_messages,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": remote_output,
                    },
                )

            remote_usage += supervisor_usage
            supervisor_messages.append(
                {"role": "assistant", "content": supervisor_response[0]}
            )
            if self.callback:
                self.callback("supervisor", supervisor_messages[-1])

            conversation_log["conversation"][-1]["output"] = supervisor_response[0]

            # Parse supervisor's decision
            if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
                try:
                    supervisor_json = json.loads(supervisor_response[0])
                except:
                    supervisor_json = _extract_json(supervisor_response[0])
            else:
                supervisor_json = _extract_json(supervisor_response[0])

            if supervisor_json["decision"] == "provide_final_answer":
                final_answer = supervisor_json["answer"]
                conversation_log["generated_final_answer"] = final_answer
                break
            else:
                next_question = supervisor_json["message"]
                worker_messages.append({"role": "user", "content": next_question})

                # Add next worker prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "local", "prompt": next_question, "output": None}
                )

        if final_answer is None:
            final_answer = "No answer found."
            conversation_log["generated_final_answer"] = final_answer

        # Add usage statistics to the log
        conversation_log["usage"]["remote"] = remote_usage.to_dict()
        conversation_log["usage"]["local"] = local_usage.to_dict()

        # Log the final result
        if logging_id:
            # use provided logging_id
            log_filename = f"{logging_id}_minion.json"
        else:
            # fall back to timestamp + task abbrev
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task = re.sub(r"[^a-zA-Z0-9]", "_", task[:15])
            log_filename = f"{timestamp}_{safe_task}.json"
        log_path = os.path.join(self.log_dir, log_filename)

        print(f"\n=== SAVING LOG TO {log_path} ===")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)

        print("\n=== MINION TASK COMPLETED ===")

        return {
            "final_answer": final_answer,
            "supervisor_messages": supervisor_messages,
            "worker_messages": worker_messages,
            "remote_usage": remote_usage,
            "local_usage": local_usage,
            "log_file": log_path,
            "conversation_log": conversation_log,
        }
