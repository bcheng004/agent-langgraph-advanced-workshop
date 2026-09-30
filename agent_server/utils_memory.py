import asyncio
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks_mason import MasonClient
from databricks_mason.memory_store import MemoryStore
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from mlflow.types.responses import ResponsesAgentRequest


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LakebaseConfig:
    instance_name: Optional[str]
    autoscaling_endpoint: Optional[str]
    autoscaling_project: Optional[str]
    autoscaling_branch: Optional[str]

    @property
    def description(self) -> str:
        return self.autoscaling_endpoint or self.instance_name or f"{self.autoscaling_project}/{self.autoscaling_branch}"


def init_lakebase_config() -> LakebaseConfig:
    """Lakebase config for the long-running server's background-task persistence.

    Agent memory no longer lives in Lakebase (see Managed Sessions / Managed Memory below), so this
    is optional: when unset, background mode is disabled but the agent still works.
    """
    endpoint = os.getenv("LAKEBASE_AUTOSCALING_ENDPOINT") or None
    raw_name = os.getenv("LAKEBASE_INSTANCE_NAME") or None
    project = os.getenv("LAKEBASE_AUTOSCALING_PROJECT") or None
    branch = os.getenv("LAKEBASE_AUTOSCALING_BRANCH") or None

    has_autoscaling = project and branch
    if not endpoint and not raw_name and not has_autoscaling:
        logger.info("No Lakebase configured - long-running background mode will be disabled.")
        return LakebaseConfig(None, None, None, None)

    # Priority: endpoint > project+branch > instance_name (mutually exclusive in the library)
    if endpoint:
        instance_name = None
        project = None
        branch = None
    elif has_autoscaling:
        instance_name = None
        endpoint = None
    else:
        instance_name = resolve_lakebase_instance_name(raw_name)
        endpoint = None
        project = None
        branch = None

    return LakebaseConfig(
        instance_name=instance_name,
        autoscaling_endpoint=endpoint,
        autoscaling_project=project,
        autoscaling_branch=branch,
    )


def _is_lakebase_hostname(value: str) -> bool:
    """Check if the value looks like a Lakebase hostname rather than an instance name."""
    # Hostname pattern: instance-{uuid}.database.{env}.cloud.databricks.com
    return ".database." in value and value.endswith(".com")


def resolve_lakebase_instance_name(
    instance_name: str, workspace_client: Optional[WorkspaceClient] = None
) -> str:
    """
    Resolve a Lakebase instance name from a hostname if needed.

    If the input is a hostname (e.g., from Databricks Apps value_from resolution),
    this will resolve it to the actual instance name by listing database instances.

    Args:
        instance_name: Either an instance name or a hostname
        workspace_client: Optional WorkspaceClient to use for resolution

    Returns:
        The resolved instance name

    Raises:
        ValueError: If the hostname cannot be resolved to an instance name
    """
    if not _is_lakebase_hostname(instance_name):
        # Input is already an instance name
        return instance_name

    # Input is a hostname - resolve to instance name
    client = workspace_client or WorkspaceClient()
    hostname = instance_name

    try:
        instances = list(client.database.list_database_instances())
    except Exception as exc:
        raise ValueError(
            f"Unable to list database instances to resolve hostname '{hostname}'. "
            "Ensure you have access to database instances."
        ) from exc

    # Find the instance that matches this hostname
    for instance in instances:
        rw_dns = getattr(instance, "read_write_dns", None)
        ro_dns = getattr(instance, "read_only_dns", None)

        if hostname in (rw_dns, ro_dns):
            resolved_name = getattr(instance, "name", None)
            if not resolved_name:
                raise ValueError(
                    f"Found matching instance for hostname '{hostname}' "
                    "but instance name is not available."
                )
            logging.info(f"Resolved Lakebase hostname '{hostname}' to instance name '{resolved_name}'")
            return resolved_name

    raise ValueError(
        f"Unable to find database instance matching hostname '{hostname}'. "
        "Ensure the hostname is correct and the instance exists."
    )


def get_user_id(request: ResponsesAgentRequest) -> Optional[str]:
    custom_inputs = dict(request.custom_inputs or {})
    if "user_id" in custom_inputs:
        return custom_inputs["user_id"]
    if request.context and getattr(request.context, "user_id", None):
        return request.context.user_id
    return None


def get_memory_store_name() -> Optional[str]:
    """Managed memory store (long-term memory). Unset disables the memory tools."""
    return os.getenv("AGENT_MEMORY_STORE") or None


def get_session_store_name() -> Optional[str]:
    """Managed session store (short-term memory). Unset falls back to an in-process checkpointer."""
    return os.getenv("AGENT_SESSION_STORE") or None


@lru_cache(maxsize=1)
def _memory_store(store_name: str) -> MemoryStore:
    return MasonClient(WorkspaceClient()).memory_stores.get(store_name)


def _memory_path(memory_key: str) -> str:
    return f"/user_memories/{memory_key.strip('/')}.md"


def _find_memory(store: MemoryStore, actor_id: str, path: str):
    return next((m for m in store.list(actor_id=actor_id, path_prefix=path) if m.path == path), None)


def memory_tools():
    """Long-term memory tools backed by Databricks Managed Memory (databricks-mason).

    ``actor_id`` comes from the trusted run config (the signed-in user), never from the model, so
    each user's memories stay partitioned. Mason's client is synchronous, so calls run in a thread.
    """
    store_name = get_memory_store_name()
    if not store_name:
        logger.warning("AGENT_MEMORY_STORE not set - long-term memory tools are disabled")
        return []

    @tool
    async def get_user_memory(query: str, config: RunnableConfig) -> str:
        """Search for relevant information about the user from long-term memory."""
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "Memory not available - no user_id provided."

        store = _memory_store(store_name)
        results = await asyncio.to_thread(store.search, actor_id=user_id, query=query, limit=5)

        if not results:
            return "No memories found for this user."

        memory_items = [f"- [{r.memory.path}]: {r.memory.content}" for r in results]
        return f"Found {len(results)} relevant memories:\n" + "\n".join(memory_items)

    @tool
    async def save_user_memory(memory_key: str, content: str, config: RunnableConfig) -> str:
        """Save information about the user to long-term memory.

        memory_key is a short slug for the topic (e.g. "preferences/language"); saving to an
        existing key overwrites it. content is the fact to remember, in plain text.
        """
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "Cannot save memory - no user_id provided."

        store = _memory_store(store_name)
        path = _memory_path(memory_key)

        def _upsert():
            if existing := _find_memory(store, user_id, path):
                existing.update(content=content)
            else:
                store.add(actor_id=user_id, path=path, content=content, description=memory_key)

        await asyncio.to_thread(_upsert)
        return f"Successfully saved memory '{memory_key}' for user."

    @tool
    async def delete_user_memory(memory_key: str, config: RunnableConfig) -> str:
        """Delete a specific memory from the user's long-term memory."""
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "Cannot delete memory - no user_id provided."

        store = _memory_store(store_name)
        path = _memory_path(memory_key)

        def _delete() -> bool:
            if existing := _find_memory(store, user_id, path):
                existing.delete()
                return True
            return False

        if not await asyncio.to_thread(_delete):
            return f"No memory found with key '{memory_key}'."
        return f"Successfully deleted memory '{memory_key}' for user."

    return [get_user_memory, save_user_memory, delete_user_memory]
