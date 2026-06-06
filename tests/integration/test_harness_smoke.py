# tests/integration/test_harness_smoke.py
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_db_conn_roundtrips_jsonb(db_conn):
    await db_conn.execute(
        """
        INSERT INTO research_journal (entry_type, title, content, structured_data)
        VALUES ('HYPOTHESIS', 't', 'c', $1)
        """,
        {"track": "trading"},  # dict → jsonb via the registered codec
    )
    row = await db_conn.fetchrow(
        "SELECT structured_data FROM research_journal WHERE title = 't'"
    )
    assert row["structured_data"] == {"track": "trading"}
