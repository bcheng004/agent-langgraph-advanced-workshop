"""Grant Lakebase Postgres permissions to a Databricks Apps service principal.

After deploying the app, run this script to grant the app's SP access to all
Lakebase schemas and tables used by the agent's memory.

Usage:
    # Interactive mode — prompts for any missing arguments:
    uv run grant-lakebase-permissions

    # Using app name (recommended — automatically resolves the SP client ID):
    uv run grant-lakebase-permissions --app-name <app-name> --memory-type <type> --instance-name <name>

    # Using explicit SP client ID:
    uv run grant-lakebase-permissions --sp-client-id <sp-client-id> --memory-type <type> --instance-name <name>

    # Autoscaling instance:
    uv run grant-lakebase-permissions --app-name <app-name> --memory-type <type> --project <project> --branch <branch>

    # Memory types: langgraph-short-term, langgraph-long-term, openai-short-term, long-running-agent
"""

import argparse
import os
import shutil
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()


# Per-memory-type schema -> table definitions.
MEMORY_TYPE_SCHEMAS: dict[str, dict[str, list[str]]] = {
    "langgraph-short-term": {
        "public": [
            "checkpoint_migrations",
            "checkpoint_writes",
            "checkpoints",
            "checkpoint_blobs",
        ],
    },
    "langgraph-long-term": {
        "public": [
            "store_migrations",
            "store",
            "store_vectors",
            "vector_migrations",
        ],
    },
    "openai-short-term": {
        "public": [
            "agent_sessions",
            "agent_messages",
        ],
    },
    "long-running-agent": {
        "agent_server": [
            "responses",
            "messages",
        ],
    },
}

# Memory types that need sequence privileges (auto-increment columns)
NEEDS_SEQUENCES = {
    "openai-short-term": ["public"],
    "long-running-agent": ["agent_server"],
}

# Shared schemas that need sequence privileges for all memory types.
# Drizzle uses __drizzle_migrations with id SERIAL PRIMARY KEY, which
# requires USAGE, SELECT, UPDATE on the backing sequence.
SHARED_SEQUENCE_SCHEMAS = ["drizzle"]

# Shared schemas granted for all memory types (chat UI persistence)
SHARED_SCHEMAS: dict[str, list[str]] = {
    "ai_chatbot": ["Chat", "Message", "User", "Vote"],
    "drizzle": ["__drizzle_migrations"],
}


def _get_databricks_profiles() -> list[dict]:
    """Get list of existing Databricks profiles."""
    if not shutil.which("databricks"):
        return []
    try:
        result = subprocess.run(
            ["databricks", "auth", "profiles"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        lines = result.stdout.strip().split("\n")
        if len(lines) <= 1:
            return []
        profiles = []
        for line in lines[1:]:
            if line.strip():
                parts = line.split()
                if parts:
                    profiles.append({"name": parts[0], "line": line})
        return profiles
    except Exception:
        return []


def _select_profile_interactive(profiles: list[dict]) -> str:
    """Let user select a profile interactively."""
    print("\nFound existing Databricks profiles:\n")
    for i, profile in enumerate(profiles, 1):
        print(f"  {i}) {profile['line']}")
    print()
    while True:
        choice = input("Enter the number of the profile you want to use: ").strip()
        if not choice:
            print("  Profile selection is required.")
            continue
        try:
            index = int(choice) - 1
            if 0 <= index < len(profiles):
                return profiles[index]["name"]
            print(f"  Please choose a number between 1 and {len(profiles)}")
        except ValueError:
            print("  Please enter a valid number.")


def _prompt_choice(prompt: str, valid: list[str]) -> str:
    """Prompt the user until they enter one of the valid choices."""
    while True:
        value = input(prompt).strip()
        if value in valid:
            return value
        print(f"  Please enter one of: {', '.join(valid)}")


def _prompt_required(prompt: str) -> str:
    """Prompt the user until they enter a non-empty value."""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("  A value is required.")


def main():
    parser = argparse.ArgumentParser(
        description="Grant Lakebase permissions to an app service principal."
    )
    parser.add_argument(
        "--profile",
        default=os.getenv("DATABRICKS_CONFIG_PROFILE"),
        help="Databricks config profile (default: DATABRICKS_CONFIG_PROFILE from .env)",
    )
    parser.add_argument(
        "--app-name",
        help="Databricks app name. The SP client ID will be resolved automatically.",
    )
    parser.add_argument(
        "--sp-client-id",
        help="Service principal client ID (UUID). Alternative to --app-name.",
    )
    parser.add_argument(
        "--memory-type",
        choices=list(MEMORY_TYPE_SCHEMAS.keys()),
        help="Memory type to grant permissions for",
    )
    parser.add_argument(
        "--instance-name",
        default=os.getenv("LAKEBASE_INSTANCE_NAME"),
        help="Lakebase instance name for provisioned instances (default: LAKEBASE_INSTANCE_NAME from .env)",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("LAKEBASE_AUTOSCALING_PROJECT"),
        help="Lakebase autoscaling project name (default: LAKEBASE_AUTOSCALING_PROJECT from .env)",
    )
    parser.add_argument(
        "--branch",
        default=os.getenv("LAKEBASE_AUTOSCALING_BRANCH"),
        help="Lakebase autoscaling branch name (default: LAKEBASE_AUTOSCALING_BRANCH from .env)",
    )
    args = parser.parse_args()

    # --- Resolve Databricks profile ---
    if not args.profile:
        profiles = _get_databricks_profiles()
        if profiles:
            args.profile = _select_profile_interactive(profiles)
            print(f"\nSelected profile: {args.profile}")
        else:
            print(
                "Error: No Databricks profiles found. Run 'databricks auth login' first.",
                file=sys.stderr,
            )
            sys.exit(1)

    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    print(f"Using Databricks profile: {args.profile}")

    # --- Resolve service principal ---
    if args.app_name and args.sp_client_id:
        print(
            "Error: Provide only one of --app-name or --sp-client-id, not both.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.app_name and not args.sp_client_id:
        print("\nHow would you like to identify the service principal?")
        print("  1) Databricks app name (recommended — auto-resolves the SP)")
        print("  2) Explicit SP client ID")
        print()
        sp_choice = _prompt_choice("Enter your choice (1 or 2): ", ["1", "2"])
        if sp_choice == "1":
            args.app_name = _prompt_required("\nEnter the Databricks app name: ")
        else:
            args.sp_client_id = _prompt_required("\nEnter the SP client ID (UUID): ")

    if args.app_name:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient(profile=args.profile)
        app = w.apps.get(name=args.app_name)
        sp_client_id = app.service_principal_client_id
        if not sp_client_id:
            print(
                f"Error: App '{args.app_name}' has no service_principal_client_id.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Resolved SP client ID from app '{args.app_name}': {sp_client_id}")
    else:
        sp_client_id = args.sp_client_id

    # --- Resolve memory type ---
    if not args.memory_type:
        memory_types = list(MEMORY_TYPE_SCHEMAS.keys())
        print("\nWhich memory type do you want to grant permissions for?")
        for i, mt in enumerate(memory_types, 1):
            print(f"  {i}) {mt}")
        print()
        mt_choice = _prompt_choice(
            f"Enter your choice (1-{len(memory_types)}): ",
            [str(i) for i in range(1, len(memory_types) + 1)],
        )
        args.memory_type = memory_types[int(mt_choice) - 1]

    # --- Resolve Lakebase connection ---
    has_provisioned = bool(args.instance_name)
    has_autoscaling = bool(args.project and args.branch)

    if not has_provisioned and not has_autoscaling:
        print("\nWhat type of Lakebase instance are you using?")
        print("  1) Autoscaling (project + branch)")
        print("  2) Provisioned (instance name)")
        print()
        lb_choice = _prompt_choice("Enter your choice (1 or 2): ", ["1", "2"])
        if lb_choice == "2":
            args.instance_name = _prompt_required(
                "\nEnter the provisioned Lakebase instance name: "
            )
        else:
            args.project = _prompt_required("\nEnter the autoscaling project name: ")
            args.branch = _prompt_required("Enter the branch name: ")
        has_provisioned = bool(args.instance_name)
        has_autoscaling = bool(args.project and args.branch)

    from databricks_ai_bridge.lakebase import (
        LakebaseClient,
        SchemaPrivilege,
        SequencePrivilege,
        TablePrivilege,
    )

    client = LakebaseClient(
        instance_name=args.instance_name or None,
        project=args.project or None,
        branch=args.branch or None,
    )
    sp_id = sp_client_id
    memory_type = args.memory_type

    if has_provisioned:
        print(f"Using provisioned instance: {args.instance_name}")
    else:
        print(f"Using autoscaling project: {args.project}, branch: {args.branch}")
    print(f"Memory type: {memory_type}")

    # Build schema -> tables map for the selected memory type
    schema_tables: dict[str, list[str]] = {
        **MEMORY_TYPE_SCHEMAS[memory_type],
        **SHARED_SCHEMAS,
    }

    # 1. Create role
    print(f"Creating role for SP {sp_id}...")
    try:
        client.create_role(sp_id, "SERVICE_PRINCIPAL")
        print("  Role created.")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  Role already exists, skipping.")
        else:
            raise

    # 2. Grant schema + table privileges
    schema_privileges = [SchemaPrivilege.USAGE, SchemaPrivilege.CREATE]
    table_privileges = [
        TablePrivilege.SELECT,
        TablePrivilege.INSERT,
        TablePrivilege.UPDATE,
        TablePrivilege.DELETE,
    ]

    for schema, tables in schema_tables.items():
        print(f"Granting schema privileges on '{schema}'...")
        try:
            client.grant_schema(
                grantee=sp_id, schemas=[schema], privileges=schema_privileges
            )
        except Exception as e:
            print(f"  Warning: schema grant failed (may not exist yet): {e}")

        qualified_tables = [f"{schema}.{t}" for t in tables]
        print(f"  Granting table privileges on {qualified_tables}...")
        try:
            client.grant_table(
                grantee=sp_id, tables=qualified_tables, privileges=table_privileges
            )
        except Exception as e:
            print(f"  Warning: table grant failed (may not exist yet): {e}")

    # 3. Grant sequence privileges (auto-increment columns).
    # Note: DELETE is not a valid privilege for sequences, so we grant only
    # USAGE, SELECT, UPDATE.
    # All memory types need drizzle sequences (Chat UI uses SERIAL PRIMARY KEY).
    # Some memory types need additional per-type sequences.
    seq_schemas = list(SHARED_SEQUENCE_SCHEMAS)
    if memory_type in NEEDS_SEQUENCES:
        seq_schemas.extend(NEEDS_SEQUENCES[memory_type])

    for schema in seq_schemas:
        print(f"Granting sequence privileges on '{schema}' schema...")
        try:
            client.grant_all_sequences_in_schema(
                grantee=sp_id,
                schemas=[schema],
                privileges=[
                    SequencePrivilege.USAGE,
                    SequencePrivilege.SELECT,
                    SequencePrivilege.UPDATE,
                ],
            )
        except Exception as e:
            print(f"  Warning: sequence grant failed (may not exist yet): {e}")

    print(
        "\nPermission grants complete. If some grants failed because tables don't "
        "exist yet, that's expected on a fresh branch — they'll be created on first "
        "agent usage. Re-run this script after the first run to grant remaining permissions."
    )


if __name__ == "__main__":
    main()
