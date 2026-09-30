"""Create the Databricks Managed Memory / Managed Sessions stores and grant an app access.

Store names default to AGENT_MEMORY_STORE / AGENT_SESSION_STORE from .env. Existing stores are
reused. Pass --app-name (after the app is deployed) to grant its service principal WRITE access.

Usage:
    uv run setup-agent-stores
    uv run setup-agent-stores --app-name <app-name>
"""

import argparse
import os

from databricks.sdk import WorkspaceClient
from databricks_mason import MasonClient
from databricks_mason.errors import AgentCliError
from dotenv import load_dotenv

load_dotenv()


_NOT_FOUND_CODES = {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}


def _get_or_create(stores, name: str, description: str):
    try:
        store = stores.get(name)
        print(f"Using existing store: {store.name}")
    except AgentCliError as e:
        if e.error_code not in _NOT_FOUND_CODES:
            raise
        store = stores.create(name, description=description)
        print(f"Created store: {store.name}")
    return store


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--memory-store", default=os.getenv("AGENT_MEMORY_STORE"))
    parser.add_argument("--session-store", default=os.getenv("AGENT_SESSION_STORE"))
    parser.add_argument("--app-name", help="Databricks App whose service principal should get WRITE access")
    args = parser.parse_args()

    if not args.memory_store or not args.session_store:
        parser.error("Set AGENT_MEMORY_STORE and AGENT_SESSION_STORE in .env or pass --memory-store/--session-store")

    w = WorkspaceClient()
    mason = MasonClient(w)

    memory_store = _get_or_create(mason.memory_stores, args.memory_store, "Agent long-term memory")
    session_store = _get_or_create(mason.session_stores, args.session_store, "Agent short-term memory")

    if args.app_name:
        sp_client_id = w.apps.get(args.app_name).service_principal_client_id
        if not sp_client_id:
            raise SystemExit(f"App '{args.app_name}' has no service_principal_client_id.")
        memory_store.grant_permission(sp_client_id, permission="WRITE")
        session_store.grant_permission(sp_client_id, permission="WRITE")
        print(f"Granted WRITE on both stores to app '{args.app_name}' SP ({sp_client_id})")


if __name__ == "__main__":
    main()
