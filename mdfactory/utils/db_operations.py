# ABOUTME: Shared database operations for push/sync modules
# ABOUTME: Contains deduplication, schema polling, and Foundry merge/upload utilities

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ..analysis.constants import SCHEMA_POLL_INTERVAL_SECONDS, SCHEMA_POLL_TIMEOUT_SECONDS
from .data_manager import (
    PLACEHOLDER_HASH,
    DataManager,
    FoundryDataSource,
    quote_sqlite_identifier,
)


def wait_for_schema(
    dataset: Any,
    expected_columns: list[str],
    timeout_seconds: int = SCHEMA_POLL_TIMEOUT_SECONDS,
    interval_seconds: int = SCHEMA_POLL_INTERVAL_SECONDS,
) -> None:
    """Wait for Foundry schema to be visible with expected columns.

    Polls the dataset until the schema matches expected columns or timeout.

    Parameters
    ----------
    dataset : Any
        Foundry dataset object with query_foundry_sql method
    expected_columns : list[str]
        List of column names that should be visible
    timeout_seconds : int
        Maximum time to wait (default: SCHEMA_POLL_TIMEOUT_SECONDS)
    interval_seconds : int
        Polling interval (default: SCHEMA_POLL_INTERVAL_SECONDS)
    """
    deadline = datetime.now(timezone.utc).timestamp() + timeout_seconds
    expected = set(expected_columns)
    last_columns = None
    while datetime.now(timezone.utc).timestamp() < deadline:
        try:
            result = dataset.query_foundry_sql("SELECT * LIMIT 0")
            last_columns = set(result.columns)
            if last_columns == expected:
                return
        except Exception:
            pass
        time.sleep(interval_seconds)
    logger.warning(f"Schema not visible after wait. Expected={expected}, last_seen={last_columns}")


def dedupe_records(
    records: list[dict[str, Any]],
    key_fields: list[str],
) -> list[dict[str, Any]]:
    """Deduplicate records by key fields, keeping the last occurrence.

    Parameters
    ----------
    records : list[dict[str, Any]]
        List of record dicts to deduplicate
    key_fields : list[str]
        Field names to use as deduplication key

    Returns
    -------
    list[dict[str, Any]]
        Deduplicated records (last occurrence wins)
    """
    if not records:
        return []
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = tuple(record.get(k) for k in key_fields)
        deduped[key] = record
    if len(deduped) < len(records):
        logger.warning(
            f"Dropped {len(records) - len(deduped)} duplicate record(s) by key {key_fields}"
        )
    return list(deduped.values())


def dedupe_dataframe(
    df: pd.DataFrame,
    key_fields: list[str],
    table_name: str,
) -> pd.DataFrame:
    """Deduplicate DataFrame rows by key fields, keeping the last occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to deduplicate
    key_fields : list[str]
        Column names to use as deduplication key
    table_name : str
        Table name for logging

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame
    """
    if df.empty:
        return df
    if any(k not in df.columns for k in key_fields):
        return df
    before = len(df)
    df = df.drop_duplicates(subset=key_fields, keep="last")
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} duplicate row(s) in {table_name}")
    return df


def drop_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    """Remove placeholder rows from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that may contain placeholder rows

    Returns
    -------
    pd.DataFrame
        DataFrame with placeholder rows removed
    """
    if df.empty or "hash" not in df.columns:
        return df
    return df[df["hash"] != PLACEHOLDER_HASH].copy()


def foundry_merge_upload(
    dm: DataManager,
    records: list[dict[str, Any]],
    key_fields: list[str],
    table_name: str,
    force: bool,
    diff: bool,
) -> int:
    """Merge and upload records to Foundry dataset.

    Handles force (overwrite) and diff (skip existing) modes for Foundry
    which requires loading all data, merging, and writing in one transaction.

    Parameters
    ----------
    dm : DataManager
        DataManager instance for the table
    records : list[dict[str, Any]]
        Records to upload
    key_fields : list[str]
        Fields used for deduplication/matching
    table_name : str
        Table name for logging
    force : bool
        If True, overwrite existing records with matching keys
    diff : bool
        If True, skip records with keys already in database

    Returns
    -------
    int
        Number of records uploaded

    Raises
    ------
    ValueError
        If duplicates exist and neither force nor diff is set
    """
    if force and diff:
        raise ValueError("Cannot use both force and diff")
    if not records:
        logger.info(f"No records to upload to {table_name}")
        return 0

    existing_df = drop_placeholder(dm.load_data())
    existing_df = dedupe_dataframe(existing_df, key_fields, table_name)
    new_df = pd.DataFrame(records)
    new_df = dedupe_dataframe(new_df, key_fields, table_name)

    existing_keys: set[tuple[Any, ...]] = set()
    if not existing_df.empty:
        existing_keys = set(existing_df[key_fields].apply(tuple, axis=1))
    new_keys = set(new_df[key_fields].apply(tuple, axis=1))

    if not force and not diff:
        overlap = existing_keys & new_keys
        if overlap:
            logger.error(f"Found {len(overlap)} duplicate key(s) in {table_name}")
            raise ValueError(
                f"{len(overlap)} key(s) already exist in {table_name}. "
                f"Use --force to overwrite or --diff to skip."
            )

    if diff:
        new_df = new_df[~new_df[key_fields].apply(tuple, axis=1).isin(existing_keys)]
        if new_df.empty:
            logger.info(f"All records already exist in {table_name}")
            return 0

    if force:
        existing_df = existing_df[~existing_df[key_fields].apply(tuple, axis=1).isin(new_keys)]

    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    dm.save_data(merged_df, overwrite=True)
    if isinstance(dm.data_source, FoundryDataSource):
        dm.data_source.wait_for_row_count(len(merged_df))
    logger.success(f"Successfully uploaded {len(new_df)} record(s) to {table_name}")
    return len(new_df)


def local_upload_with_modes(
    dm: DataManager,
    records: list[dict[str, Any]],
    key_fields: list[str],
    table_name: str,
    force: bool,
    diff: bool,
) -> int:
    """Upload records to a local table (SQLite or CSV) with force/diff/default behavior.

    Parameters
    ----------
    dm : DataManager
        DataManager instance for the target table
    records : list[dict[str, Any]]
        Records to upload
    key_fields : list[str]
        Key fields used for duplicate detection and overwrite logic
    table_name : str
        Table name for logging and error messages
    force : bool
        If True, delete existing rows matching keys before insert
    diff : bool
        If True, skip rows that already exist by key

    Returns
    -------
    int
        Number of records inserted

    Raises
    ------
    ValueError
        If duplicate keys are found in default mode (no force/diff)
    """
    if force and diff:
        raise ValueError("Cannot use both force and diff")
    if not records:
        logger.info(f"No records to upload to {table_name}")
        return 0

    existing_df = drop_placeholder(dm.load_data())
    existing_df = dedupe_dataframe(existing_df, key_fields, table_name)
    existing_keys: set[tuple[Any, ...]] = set()
    if not existing_df.empty and all(k in existing_df.columns for k in key_fields):
        existing_keys = set(existing_df[key_fields].apply(tuple, axis=1))

    def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(record.get(k) for k in key_fields)

    if diff:
        original_count = len(records)
        records = [record for record in records if record_key(record) not in existing_keys]
        skipped_count = original_count - len(records)
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} existing record(s) in {table_name} (diff mode)")
        if not records:
            logger.info(f"All records already exist in {table_name}")
            return 0

    if force:
        logger.info(f"Force mode: deleting existing records for {len(records)} key(s)")
        for record in records:
            conditions = {key: record.get(key) for key in key_fields}
            try:
                dm.delete_data(conditions)
            except Exception as e:
                logger.warning(f"Failed to delete record with key {record_key(record)}: {e}")

    if not force and not diff:
        new_keys = {record_key(record) for record in records}
        overlap = existing_keys & new_keys
        if overlap:
            logger.error(f"Found {len(overlap)} duplicate key(s) in {table_name}")
            raise ValueError(
                f"{len(overlap)} key(s) already exist in {table_name}. "
                f"Use --force to overwrite or --diff to skip."
            )

    logger.info(f"Uploading {len(records)} record(s) to {table_name}...")
    dm.save_data(records, overwrite=False)
    logger.success(f"Successfully uploaded {len(records)} record(s) to {table_name}")
    return len(records)


def upload_records(
    records: list[dict[str, Any]],
    table_name: str,
    key_fields: list[str],
    force: bool = False,
    diff: bool = False,
) -> int:
    """Upload records to database with deduplication and backend dispatch.

    Deduplicates input records, then routes to the appropriate backend
    (Foundry or SQLite) with the specified conflict-resolution mode.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Records to upload
    table_name : str
        Target table name
    key_fields : list[str]
        Fields used for duplicate detection
    force : bool
        If True, overwrite existing records with matching keys
    diff : bool
        If True, skip records with keys already in database

    Returns
    -------
    int
        Number of records uploaded

    Raises
    ------
    ValueError
        If duplicates exist and neither force nor diff is set,
        or if both force and diff are set
    """
    if force and diff:
        raise ValueError("Cannot use both force and diff")
    records = dedupe_records(records, key_fields)
    if not records:
        logger.info(f"No records to upload to {table_name}")
        return 0

    dm = DataManager(table_name)
    if isinstance(dm.data_source, FoundryDataSource):
        return foundry_merge_upload(
            dm=dm,
            records=records,
            key_fields=key_fields,
            table_name=table_name,
            force=force,
            diff=diff,
        )
    return local_upload_with_modes(
        dm=dm,
        records=records,
        key_fields=key_fields,
        table_name=table_name,
        force=force,
        diff=diff,
    )


def init_sqlite_tables(
    tables_to_init: list[tuple[str, dict, list[str]]],
    reset: bool = False,
) -> dict[str, bool]:
    """Initialize SQLite database tables with schema via placeholder records.

    Creates the database file if needed, then creates each table by inserting
    and removing a placeholder record (mirroring the Foundry initialization
    pattern). With reset=True, drops existing tables before recreating them.

    Parameters
    ----------
    tables_to_init : list[tuple[str, dict, list[str]]]
        List of (table_name, placeholder_record, columns) to initialize
    reset : bool
        Drop and recreate tables that already exist

    Returns
    -------
    dict[str, bool]
        {table_name: was_created}
    """
    if not tables_to_init:
        return {}

    from ..settings import Settings

    # Resolve the db_path from the first table name
    config = Settings()
    first_table_name = tables_to_init[0][0]
    db_path = Path(config.get_db_path(first_table_name, "sqlite"))

    # Ensure the database file exists with proper pragmas
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path, autocommit=True) as con:
        con.execute("PRAGMA user_version = 1")
        con.execute("PRAGMA journal_mode=WAL")

    results: dict[str, bool] = {}
    created_count = 0
    existing_count = 0

    for table_name, placeholder, _columns in tables_to_init:
        # Check if table already exists via raw SQL (DataManager may not be
        # available yet if the db was just created)
        with sqlite3.connect(db_path, autocommit=True) as con:
            count = con.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()[0]
            table_already_exists = count > 0

        if table_already_exists and not reset:
            logger.debug(f"Table already exists: {table_name}")
            results[table_name] = False
            existing_count += 1
            continue

        if table_already_exists and reset:
            quoted_table_name = quote_sqlite_identifier(table_name)
            with sqlite3.connect(db_path, autocommit=True) as con:
                con.execute(f"DROP TABLE IF EXISTS {quoted_table_name}")
            logger.debug(f"Dropped existing table: {table_name}")

        # Create table by saving placeholder, then removing it
        dm = DataManager(table_name)
        dm.save_data(placeholder)
        dm.delete_data({"hash": PLACEHOLDER_HASH})

        results[table_name] = True
        created_count += 1

    if created_count == 0:
        logger.info(f"All {existing_count} table(s) already initialized")
    else:
        logger.success(
            f"Initialization complete: {created_count} created, {existing_count} already existed"
        )

    return results


def init_csv_tables(
    tables_to_init: list[tuple[str, dict, list[str]]],
    reset: bool = False,
) -> dict[str, bool]:
    """Initialize CSV file tables with headers.

    Creates CSV files for each table with column headers only.
    With reset=True, overwrites existing files.

    Parameters
    ----------
    tables_to_init : list[tuple[str, dict, list[str]]]
        List of (table_name, placeholder_record, columns) to initialize
    reset : bool
        Overwrite existing CSV files

    Returns
    -------
    dict[str, bool]
        {table_name: was_created}
    """
    if not tables_to_init:
        return {}

    from ..settings import Settings

    config = Settings()
    results: dict[str, bool] = {}
    created_count = 0
    existing_count = 0

    for table_name, _placeholder, columns in tables_to_init:
        csv_path = Path(config.get_csv_path(table_name))

        if csv_path.exists() and not reset:
            logger.debug(f"CSV file already exists: {csv_path}")
            results[table_name] = False
            existing_count += 1
            continue

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
        logger.debug(f"Created CSV file: {csv_path}")

        results[table_name] = True
        created_count += 1

    if created_count == 0:
        logger.info(f"All {existing_count} CSV file(s) already initialized")
    else:
        logger.success(
            f"Initialization complete: {created_count} created, {existing_count} already existed"
        )

    return results


def query_existing_hashes(table_name: str) -> set[str]:
    """Query database for all existing hashes in a table (excludes placeholder).

    Parameters
    ----------
    table_name : str
        Table name to query

    Returns
    -------
    set[str]
        Set of hash values currently in table
    """
    try:
        dm = DataManager(table_name)
        df = dm.load_data()
        if df.empty or "hash" not in df.columns:
            return set()
        hashes = set(df["hash"].unique())
        hashes.discard(PLACEHOLDER_HASH)
        return hashes
    except FileNotFoundError:
        return set()


def init_foundry_tables(
    tables_to_init: list[tuple[str, dict, list[str]]],
    reset: bool = False,
) -> dict[str, bool]:
    """Initialize Foundry dataset tables.

    Creates Foundry datasets with placeholder records to establish schemas.

    Parameters
    ----------
    tables_to_init : list[tuple[str, dict, list[str]]]
        List of (table_name, placeholder_record, columns) to initialize
    reset : bool
        Recreate tables even if they exist

    Returns
    -------
    dict[str, bool]
        {table_name: was_created}
    """
    import os
    import tempfile
    from uuid import uuid4

    from foundry_dev_tools import FoundryContext

    from ..settings import Settings

    config = Settings()
    ctx = FoundryContext()
    results = {}

    # Fail fast on global auth/connectivity issues instead of retrying every table.
    try:
        if hasattr(ctx, "multipass") and hasattr(ctx.multipass, "get_user_info"):
            ctx.multipass.get_user_info()
    except Exception as e:
        raise ValueError(f"Foundry connectivity/authentication check failed: {e}") from e

    created_count = 0
    existing_count = 0
    failed_count = 0
    failures: list[str] = []

    for table_name, placeholder, columns in tables_to_init:
        dataset_path = config.get_foundry_path(table_name)
        try:
            dataset_exists = FoundryDataSource.dataset_exists(dataset_path)
            if dataset_exists:
                try:
                    dataset = ctx.get_dataset_by_path(dataset_path, create_if_not_exist=False)
                    result = dataset.query_foundry_sql("SELECT * LIMIT 0")
                    existing_columns = set(result.columns)
                    expected_columns = set(columns)

                    if existing_columns == expected_columns and not reset:
                        logger.debug(f"Dataset already initialized: {dataset_path}")
                        results[table_name] = False
                        existing_count += 1
                        continue
                    if existing_columns == expected_columns and reset:
                        logger.info(f"Reset mode: recreating dataset {dataset_path}")
                    if existing_columns != expected_columns and not reset:
                        missing = expected_columns - existing_columns
                        extra = existing_columns - expected_columns
                        logger.error(f"Schema mismatch in {table_name}")
                        if missing:
                            logger.error(f"  Missing columns: {missing}")
                        if extra:
                            logger.error(f"  Extra columns: {extra}")
                        failures.append(
                            f"{table_name}: schema mismatch "
                            f"(missing={sorted(missing)}, extra={sorted(extra)})"
                        )
                        results[table_name] = False
                        failed_count += 1
                        continue
                except Exception as e:
                    logger.warning(
                        f"Schema check failed for {table_name}; initializing dataset. Error: {e}"
                    )

            dataset = ctx.get_dataset_by_path(
                dataset_path,
                create_if_not_exist=True,
                create_branch_if_not_exists=True,
            )

            if dataset_exists and not reset:
                logger.info(f"Dataset exists without schema, initializing: {dataset_path}")
            elif dataset_exists and reset:
                logger.info(f"Reset mode: reinitializing dataset: {dataset_path}")

            placeholder_df = pd.DataFrame([placeholder])
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
                placeholder_df.to_parquet(tmp_file.name, index=False)
                dataset.upload_file(
                    file_path=Path(tmp_file.name),
                    path_in_foundry_dataset=f"init_{uuid4()}.parquet",
                    transaction_type="SNAPSHOT",
                )
                os.unlink(tmp_file.name)

            schema = dataset.infer_schema()
            logger.debug(f"Inferred schema for {table_name}: {schema}")
            dataset.start_transaction(start_transaction_type="UPDATE")
            transaction_rid = dataset.transaction.get("rid")
            logger.debug(f"Uploading schema to transaction {transaction_rid}")
            dataset.upload_schema(transaction_rid, schema)
            dataset.commit_transaction()
            logger.debug(f"Schema transaction committed for {table_name}")

            wait_for_schema(dataset, columns)
            try:
                dm = DataManager(table_name)
                dm.delete_data({"hash": PLACEHOLDER_HASH})
                logger.info("Removed placeholder row after schema initialization")
            except Exception as e:
                logger.warning(f"Failed to remove placeholder row: {e}")

            logger.info(f"Initialized with schema: {dataset_path}")
            results[table_name] = True
            created_count += 1

        except Exception as e:
            logger.error(f"Failed to initialize {dataset_path}: {e}")
            failures.append(f"{table_name}: {e}")
            results[table_name] = False
            failed_count += 1

    # Summary
    if failed_count > 0:
        logger.warning(
            f"Initialization: {created_count} created, {existing_count} already existed, "
            f"{failed_count} failed"
        )
        failure_summary = "\n".join(failures)
        raise ValueError(f"Foundry initialization failed:\n{failure_summary}")
    elif created_count == 0:
        logger.info(f"All {existing_count} datasets already initialized")
    else:
        logger.success(
            f"Initialization complete: {created_count} created, {existing_count} already existed"
        )

    return results
