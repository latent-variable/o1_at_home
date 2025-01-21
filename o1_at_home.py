"""
title: Think-Respond Chain Pipe, o1 at home
author: latent-variable
github: https://github.com/latent-variable/o1_at_home
open-webui: https://openwebui.com/f/latentvariable/o1_at_home/
Blog post: https://o1-at-home.hashnode.dev/run-o1-at-home-privately-think-respond-pipe-tutorial-with-open-webui-ollama
version: 0.5.2
Descrition: Think-Respond pipe that has an internal reasoning steps and another for producing a final response based on the reasoning.
            Now supports openAI api along with ollama, you can mix and match models 

Instructions: 
To use the o1 at home pipe, follow these steps:

Add the Pipe Manifold:
Navigate to the Admin Panel and add the pipe to the list of available "Functions" using the '+'.
This is not a "pipeline", Ensure you are using Function tab. 
If you are copying the code you might need to give it name and descriprition 

Enable the Pipe Manifold:
After adding it, enable the pipe to make it active.

Customize Settings:
Use the configuration menu (accessed via the settings cog) to tailor the pipeline to your needs:
    Select Models: Choose your desired thinking models and response model.
    Show Reasoning: Decide whether to display the reasoning process or keep it hidden.
    Set Thinking Time: Specify the maximum time allowed for the reasoning model to process.
Save and Apply:
Once configured, save your settings to apply the changes.
You should now have o1 at home in your dorp down. 

These steps ensure the pipe is set up correctly and functions according to your requirements.
"""

import json
from time import time
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Awaitable, Any, AsyncGenerator
import asyncio
from fastapi import Request
from open_webui.utils.misc import get_last_user_message
from open_webui.main import chat_completion
from open_webui.routers.ollama import generate_chat_completion as ollama_chat_completion
from open_webui.routers.openai import generate_chat_completion as openai_chat_completion
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.set_name("think_respond_chain_pipe")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


@dataclass
class User:
    id: str
    email: str
    name: str
    role: str


class Pipe:
    class Valves(BaseModel):
        THINKING_MODEL: str = Field(
            default="your_thinking_model_id_here",
            description="Model used for the internal reasoning step. Separate multiple models with a comma.",
        )
        USE_OPENAI_API_THINKING_MODEL: bool = Field(
            default=False,
            description="Off will use Ollama, On will use any OpenAI API",
        )
        RESPONDING_MODEL: str = Field(
            default="your_responding_model_id_here",
            description="Model used for producing the final response.",
        )
        USE_OPENAI_API_RESPONDING_MODEL: bool = Field(
            default=False,
            description="Off will use Ollama, On will use any OpenAI API",
        )
        ENABLE_SHOW_THINKING_TRACE: bool = Field(
            default=False,
            description="Toggle show thinking trace.",
        )

        MAX_THINKING_TIME: int = Field(
            default=120,
            description="Maximum time in seconds that each thinking model is allowed to run for.",
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.total_thinking_tokens = 0
        self.max_thinking_time_reached = False
        self.__user__ = None
        self._json_buffer = ""
    

    def pipes(self):
        name = "o1-"
        for model in self.valves.THINKING_MODEL.split(","):
            name += model.strip().split(":")[0] + "-"
        name = name[:-1] + "-to-" + self.valves.RESPONDING_MODEL.strip().split(":")[0]
        return [{"name": name, "id": name}]

    def get_chunk_content(self, chunk: bytes):
        """
        Accumulate chunk data in a buffer and extract complete JSON objects
        from the buffer.
        """
        self._json_buffer += chunk.decode("utf-8")

        # Attempt to parse out valid JSON lines in a loop
        while True:
            # Each line might end with "\n", or the next JSON object
            # might just start right after the prior one. So look for
            # a new line or some other marker.
            newline_index = self._json_buffer.find("\n")
            if newline_index == -1:
                # No complete line yet; wait for more data
                break

            line = self._json_buffer[:newline_index].strip()
            self._json_buffer = self._json_buffer[newline_index + 1:]

            if not line:
                continue

            try:
                chunk_data = json.loads(line)
                if "message" in chunk_data and "content" in chunk_data["message"]:
                    yield chunk_data["message"]["content"]
                if chunk_data.get("done", False):
                    break
            except json.JSONDecodeError as e:
                logger.error(
                    f'ChunkDecodeError: unable to parse "{line[:100]}": {e}'
                )
                # If we get a decode error, it likely means
                # this line is incomplete or malformed. You can:
                # 1. Keep a separate "incomplete_line" buffer and
                #    re-append it to _json_buffer for the next iteration, or
                # 2. Simply ignore malformed lines if you expect partial data.
                #
                # A simple approach is to re-append the line to the buffer:
                self._json_buffer = line + "\n" + self._json_buffer
                break

    async def get_response(
        self, model: str, messages: List[Dict[str, str]], thinking: bool, stream: bool
    ):
        """
        Generate a response from the appropriate API based on the provided flags.

        Args:
            model (str): The model ID to use for the API request.
            messages (List[Dict[str, str]]): The list of messages for the API to process.
            thinking (bool): Whether this is the 'thinking' phase or the 'responding' phase.

        Returns:
            tuple: (response, api_source) where `response` is the API response object
                and `api_source` is a string ('openai' or 'ollama') indicating the API used.
        """
        # Determine which API to use based on the `thinking` flag and the corresponding valve
        use_openai_api = (
            self.valves.USE_OPENAI_API_THINKING_MODEL
            if thinking
            else self.valves.USE_OPENAI_API_RESPONDING_MODEL
        )

        # Select the appropriate API and identify the source
        if use_openai_api:
            generate_completion = openai_chat_completion
        else:
            generate_completion = ollama_chat_completion

        # Generate response
        response = await generate_completion( self.__request__, 
            {"model": model, "messages": messages, "stream": stream}, user=self.__user__, 
        )

        return response

    async def get_completion(
        self,
        model: str,
        messages: list,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ):
        response = None
        try:
            thinking = False
            stream = False
            response = await self.get_response(model, messages, thinking, stream)

            if not response:
                return "**No content available**"

            # --- Handle old format (OpenAI style: { "choices": [ { "message": {...} } ] })
            if "choices" in response:
                if not response["choices"]:
                    return "**No content available**"
                return response["choices"][0]["message"]["content"]

            # --- Handle new format (phi4 style: { "message": { "content": ... }, "done_reason": ..., "done": ... })
            if "message" in response and "content" in response["message"]:
                return response["message"]["content"]

            # If neither format is recognized
            return "**No content available**"

        except Exception as e:
            await self.set_status_end(f"Error: Is {model} a valid model? ({e})", __event_emitter__)
        finally:
            if response and hasattr(response, "close"):
                await response.close()

    async def stream_response(
        self,
        model: str,
        messages: List[Dict[str, str]],
        thinking: bool,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> AsyncGenerator[str, None]:

        start_thought_time = time()
        try:
            stream = True
            response = await self.get_response(model, messages, thinking, stream)
            while True:
                chunk = await response.body_iterator.read(1024)
                if not chunk:  # No more data
                    break
                for part in self.get_chunk_content(chunk):
                    yield part

                if thinking:
                    current_time = (
                        time()
                    )  # check to see if thought time has been exceded
                    if (
                        current_time - start_thought_time
                    ) > self.valves.MAX_THINKING_TIME:
                        logger.info(
                            f'Max thinking Time reached in stream_response of thinking model "'
                        )
                        self.max_thinking_time_reached = True
                        break
            
            # Force-close the stream after breaking:
            if self.max_thinking_time_reached:
                await response.close()
                return

        except Exception as e:
            if thinking:
                api = 'openai' if self.valves.USE_OPENAI_API_THINKING_MODEL else 'Ollama'
                category = 'Thinking'
            else:
                api = 'OpenAI' if self.valves.USE_OPENAI_API_RESPONDING_MODEL else 'Ollama'
                category = 'Responding'
            await self.set_status_end(f"{category} Error: ensure {model} is a valid model option in the {api} api {e}", __event_emitter__)
        finally:
            # Always close if response is still open
            if response and hasattr(response, "close"):
                await response.close()

    async def run_step(
        self,
        model: str,
        messages: list,
        prompt: str,
        thinking: bool,
        step_name: str,
        title_name: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        messages = json.loads(json.dumps(messages))
        messages[-1] = {
            "role": "user",
            "content": prompt,
        }
        
        await self.send_data("\n### "+title_name+"\n", thinking, __event_emitter__)

        response_text = ""
        num_tokens = 0
        async for chunk in self.stream_response(
            model.strip(), messages, thinking, __event_emitter__
        ):
            response_text += chunk
            num_tokens += 1
            await self.send_data(chunk, thinking, __event_emitter__)
            await self.set_status(f"{step_name} ({num_tokens} tokens)", __event_emitter__)
        if thinking:
            self.total_thinking_tokens += num_tokens
        return response_text.strip()

    async def run_thinking(
        self,
        k: int,
        n: int,
        model: str,
        messages: list,
        query: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        # Deep copy the messages to avoid changing the original
        thinking_with = ""
        if n == 1:
            thinking_with = f"with {model}"
        else:
            thinking_with = f"with {model} {k}/{n}"

        prompt = "You are a reasoning model.\n"
        prompt += "Think carefully about the user's request and output your reasoning steps.\n"
        prompt += "Do not answer the user directly, just produce a hidden reasoning chain.\n"
        prompt += "First rephrase the user prompt, then answer using multiple thinking-path to give all possible answers.\n"
        prompt += f"User Query: {query}"

        reasoning = await self.run_step(
            model, messages, prompt, True, f"Thinking {thinking_with}", f"`{model}` thoughts", __event_emitter__
        )

        await self.set_status(f"Finished thinking {thinking_with}", __event_emitter__)

        await asyncio.sleep(0.2)
        return reasoning

    async def run_responding(
        self,
        messages: list,
        query: str,
        reasonings: list,
        is_final_step: bool,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        await self.set_status("Formulating response...", __event_emitter__)

        prompt = "Here is some internal reasoning to guide your response:\n"
        prompt += f"<reasoning>{reasonings[0]}<reasoning-end>\n"
        for reasoning in reasonings[1:]:
            prompt += "Here is some other internal reasoning to guide your response:\n"
            prompt += f"<reasoning>{reasoning}<reasoning-end>\n"
        prompt += f"Use this reasoning to respond in concise and helpful manner to the user's query: {query}"

        response_text = await self.run_step(
            self.valves.RESPONDING_MODEL.strip(), messages, prompt, not is_final_step, "Generating response", "Response", __event_emitter__
        )

        await asyncio.sleep(0.2)
        return response_text
    
    async def run_thinking_pipeline(
        self,
        k: int,
        models: list,
        messages: list,
        query: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        response = await self.run_thinking(k + 1, len(models), models[k], messages, query, __event_emitter__)

        # If you want to implement some custom logic after the initial thoughts, you can do so here
        # For instance, you could implement reflections or CoT (Chain of Thought) here
        
        return response
        
    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]],
        __request__: Request,
        __task__=None,
    ) -> str:

        print(__event_emitter__)
        # Get relavant info
        self.__user__ = User(**__user__)
        self.__request__ = __request__
        messages = body["messages"]
        query = get_last_user_message(messages)

        if (
            __task__ == None
        ):  # only perform thinking when not a defined task like title generation
            # Run the "thinking" step
            # Clone the messages to avoid changing the original
            tik = time()
            models = self.valves.THINKING_MODEL.split(",")
            reasonings = [await self.run_thinking_pipeline(model, models, messages, query, __event_emitter__) for model in range(len(models))]
            total_thought_duration = int(time() - tik)

            # Run the "responding" step using the reasoning
            await self.run_responding(messages, query, reasonings, True, __event_emitter__)

            if self.max_thinking_time_reached:
                await self.set_status_end(f"Thought for {self.total_thinking_tokens} tokens in max allowed time of {total_thought_duration} seconds", __event_emitter__)
            else:
                await self.set_status_end(f"Thought for only {self.total_thinking_tokens} tokens in {total_thought_duration} seconds", __event_emitter__)
            return ""
        else:
            # avoid thinking and just return a regular response or named task, like tags
            message = await self.get_completion(
                self.valves.RESPONDING_MODEL.strip(), messages, __event_emitter__
            )
            return message

    async def set_status(self, description: str, __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None):
        await __event_emitter__({"type": "status", "data": {"description": description, "done": False}})

    async def send_data(self, data: str, thinking: bool, __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None):
        if not thinking or self.valves.ENABLE_SHOW_THINKING_TRACE:
            await __event_emitter__({"type": "message", "data": {"content": data, "role": "assistant-thinking" if thinking else "assistant"}})
    
    async def set_status_end(self, data: str, __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None):
        await __event_emitter__({"type": "status", "data": {"description": data, "done": True}})
            