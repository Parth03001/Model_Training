"""Cross-database compatible column types.

SQLite doesn't support PostgreSQL UUID or native Enum types.
This module provides a UUID column type that works with both.
"""

import uuid

from sqlalchemy import String, TypeDecorator


class UUIDType(TypeDecorator):
    """Platform-independent UUID type.

    Uses PostgreSQL's UUID type when available, otherwise stores as String(36).
    """
    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return uuid.UUID(value) if not isinstance(value, uuid.UUID) else value
        return value
