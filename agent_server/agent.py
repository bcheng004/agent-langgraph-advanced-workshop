import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Optional, Sequence, TypedDict

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from databricks_mason.langgraph import checkpointer, thread_config
from langchain.agents import create_agent
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)
from typing_extensions import Annotated

from agent_server.prompts import SYSTEM_PROMPT
from agent_server.utils import (
    _get_or_create_thread_id,
    get_user_workspace_client,
    init_mcp_client,
    process_agent_astream_events,
)
from agent_server.utils_memory import (
    get_session_store_name,
    get_user_id,
    memory_tools,
)

logger = logging.getLogger(__name__)
mlflow.langchain.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
sp_workspace_client = WorkspaceClient()

LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-5"


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


class StatefulAgentState(TypedDict, total=False):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    custom_inputs: dict[str, Any]
    custom_outputs: dict[str, Any]


async def init_agent(workspace_client: Optional[WorkspaceClient] = None):
    tools = [get_current_time] + memory_tools()
    # To use MCP server tools instead, uncomment the below lines:
    # mcp_client = init_mcp_client(workspace_client or sp_workspace_client)
    # try:
    #     tools.extend(await mcp_client.get_tools())
    # except Exception:
    #     logger.warning("Failed to fetch MCP tools. Continuing without MCP tools.", exc_info=True)

    model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        # Short-term memory: Databricks Managed Sessions via databricks-mason. Persists LangGraph
        # checkpoints to AGENT_SESSION_STORE; falls back to an in-process InMemorySaver when unset.
        checkpointer=checkpointer(get_session_store_name()),
        state_schema=StatefulAgentState,
    )


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]

    custom_outputs: dict[str, Any] = {}
    if user_id := get_user_id(request):
        custom_outputs["user_id"] = user_id
    return ResponsesAgentResponse(output=outputs, custom_outputs=custom_outputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    thread_id = _get_or_create_thread_id(request)
    mlflow.update_current_trace(metadata={"mlflow.trace.session": thread_id})

    user_id = get_user_id(request)
    if not user_id:
        logger.warning("No user_id provided - memory features will not be available")

    # actor_id partitions sessions per user in the managed session store (defaults to thread_id).
    config: dict[str, Any] = thread_config(thread_id, actor=user_id)
    if user_id:
        config["configurable"]["user_id"] = user_id

    input_state: dict[str, Any] = {
        "messages": to_chat_completions_input([i.model_dump() for i in request.input]),
        "custom_inputs": dict(request.custom_inputs or {}),
    }

    # By default, uses service principal credentials.
    # For on-behalf-of user authentication, pass get_user_workspace_client() to init_agent.
    agent = await init_agent()

    async for event in process_agent_astream_events(
        agent.astream(input_state, config, stream_mode=["updates", "messages"])
    ):
        yield event
