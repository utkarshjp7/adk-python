# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
from typing import Any
from typing import Union

import agent
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.genai import types
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import export
from opentelemetry.sdk.trace import TracerProvider

load_dotenv(override=True)

APP_NAME = "human_in_the_loop"
USER_ID = "1234"
SESSION_ID = "session1234"

session_service = InMemorySessionService()


async def main():
  session = await session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
  )
  runner = Runner(
      agent=agent.root_agent,
      app_name=APP_NAME,
      session_service=session_service,
  )

  print(f"--- Started session {session.id} ---")

  async def call_agent(query: str):
    content = types.Content(role="user", parts=[types.Part(text=query)])

    print(f'\n>>> User Query: "{query}"\n')

    events_async = runner.run_async(
        session_id=session.id, user_id=USER_ID, new_message=content
    )

    long_running_function_call: Union[types.FunctionCall, None] = None
    initial_tool_response: Union[types.FunctionResponse, None] = None
    ticket_id: Union[str, None] = None

    async for event in events_async:
      if event.content and event.content.parts:
        for i, part in enumerate(event.content.parts):
          if part.text:
            print(f"\n>>> Agent: {part.text.strip()}\n")
          if part.function_call:
            if not long_running_function_call and part.function_call.id in (
                event.long_running_tool_ids or []
            ):
              long_running_function_call = part.function_call
          if part.function_response:
            if (
                long_running_function_call
                and part.function_response.id == long_running_function_call.id
            ):
              initial_tool_response = part.function_response
              if initial_tool_response.response:
                ticket_id = initial_tool_response.response.get("ticketId")

    if (
        long_running_function_call
        and initial_tool_response
        and initial_tool_response.response.get("status") == "pending"
    ):
      print(f"--- Sending ticket for manager approval with ID: {ticket_id}, and waiting 10 seconds to simulate manager approval.---\n")
      await asyncio.sleep(10)

      updated_tool_output_data = {
          "status": "approved",
          "ticketId": ticket_id,
          "approver_feedback": "Approved by manager at " + str(
              asyncio.get_event_loop().time()
          ),
      }

      updated_function_response_part = types.Part(
          function_response=types.FunctionResponse(
              id=long_running_function_call.id,
              name=long_running_function_call.name,
              response=updated_tool_output_data,
          )
      )

      print(f"--- Manager approved ticket {ticket_id}. ---")
      print(f"--- Sending update to agent for call ID: {long_running_function_call.id}. ---")

      async for event in runner.run_async(
          session_id=session.id,
          user_id=USER_ID,
          new_message=types.Content(
              parts=[updated_function_response_part], role="user"
          ),
      ):
        if event.content and event.content.parts:
          for i, part in enumerate(event.content.parts):
            if part.text:
              print(f"\n>>> Agent: {part.text.strip()}\n")

    elif long_running_function_call and not initial_tool_response:
      print(
          f"--- Long running function '{long_running_function_call.name}' was"
          " called, but its initial response was not captured. ---"
      )
    elif not long_running_function_call:
      print(
          "--- No long running function call was detected in the initial"
          " turn. ---"
      )

  print("--- Sending first request ---")
  task1 = asyncio.create_task(call_agent("Please reimburse $200 for team lunch"))
  print("--- Waiting  seconds before sending second request. ---")
  await asyncio.sleep(2)
  print("--- Sending second request ---")
  task2 = asyncio.create_task(call_agent("Please reimburse $300 for conference travel"))

  await task1
  await task2


if __name__ == "__main__":
  provider = TracerProvider()
  project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
  if not project_id:
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set.")

  asyncio.run(main())

  provider.force_flush()
