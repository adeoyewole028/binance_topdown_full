"""User-selected D07 targets. External activation is a separate gate."""
from dataclasses import dataclass


@dataclass(frozen=True)
class OperationsPolicy:
    scan_interval: int = 300
    scan_health_timeout: int = 600
    price_refresh_interval: int = 30
    price_expiry: int = 60
    reconciliation_interval: int = 15
    backup_retention_days: int = 7
    recovery_point_seconds: int = 900
    recovery_time_seconds: int = 3600
