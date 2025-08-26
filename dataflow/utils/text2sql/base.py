from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataflow import get_logger


# ============== Base Data Classes ==============
@dataclass
class DatabaseInfo:
    """Database connection information"""
    db_id: str
    db_type: str
    connection_info: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QueryResult:
    """Standard query result format"""
    success: bool
    data: List[Dict[str, Any]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    error: Optional[str] = None
    row_count: int = 0

# ============== Base Database Connector ==============
class DatabaseConnectorABC(ABC):
    
    def __init__(self):
        self.logger = get_logger()
    
    @abstractmethod
    def connect(self, connection_info: Dict) -> Any:
        """Create database connection"""
        pass
    
    @abstractmethod
    def execute_query(self, connection: Any, sql: str, params: Optional[Tuple] = None) -> QueryResult:
        """Execute SQL query and return standardized result"""
        pass
    
    @abstractmethod
    def get_schema_info(self, connection: Any) -> Dict[str, Any]:
        """Get complete database schema information"""
        pass

    @abstractmethod
    def discover_databases(self, config: Dict) -> Dict[str, DatabaseInfo]:
        """Discover available databases"""
        pass
    
    def validate_connection(self, connection: Any) -> bool:
        """Check if connection is still valid"""
        try:
            self.execute_query(connection, "SELECT 1")
            return True
        except Exception as e:
            self.logger.debug(f"Connection validation failed: {e}")
            return False
    
    def close(self, connection: Any):
        """Close database connection"""
        try:
            if hasattr(connection, 'close'):
                connection.close()
        except Exception as e:
            self.logger.debug(f"Error closing connection: {e}")