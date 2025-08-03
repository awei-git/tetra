from typing import List, Tuple, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)


class QueryExecutor:
    """Service for safely executing SQL queries"""
    
    # Patterns for unsafe SQL
    UNSAFE_PATTERNS = [
        r'\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXEC|EXECUTE)\b',
        r'\b(GRANT|REVOKE|COMMIT|ROLLBACK|SAVEPOINT)\b',
        r';\s*\S',  # Multiple statements (semicolon followed by non-whitespace)
    ]
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def is_safe_query(self, sql: str) -> bool:
        """Check if SQL query is safe (read-only)"""
        if not sql or len(sql.strip()) < 10:  # Too short to be valid
            logger.warning(f"SQL too short: {sql}")
            return False
            
        sql_upper = sql.upper()
        
        # Check for unsafe patterns
        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                return False
        
        # Must start with SELECT or WITH
        sql_stripped = sql_upper.strip()
        if not (sql_stripped.startswith('SELECT') or sql_stripped.startswith('WITH')):
            return False
        
        # Must have FROM clause
        if 'FROM' not in sql_upper:
            logger.warning(f"SQL missing FROM clause: {sql}")
            return False
        
        return True
    
    async def execute_query(
        self, 
        sql: str, 
        params: List[Any] = None
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Execute SQL query and return results with column names"""
        
        if not self.is_safe_query(sql):
            raise ValueError("Unsafe SQL query detected")
        
        try:
            # Execute query
            if params:
                rows = await self.db.fetch(sql, *params)
            else:
                rows = await self.db.fetch(sql)
            
            # Get column names
            if rows:
                columns = list(rows[0].keys())
                # Convert rows to list of dicts
                results = [dict(row) for row in rows]
            else:
                # Try to get columns from query preparation
                stmt = await self.db.prepare(sql)
                columns = [attr.name for attr in stmt.get_attributes()]
                results = []
            
            return results, columns
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    async def estimate_query_cost(self, sql: str) -> Dict[str, Any]:
        """Estimate query cost using EXPLAIN"""
        try:
            explain_sql = f"EXPLAIN (FORMAT JSON, ANALYZE FALSE) {sql}"
            result = await self.db.fetchval(explain_sql)
            return {
                "estimated_cost": result[0]["Plan"]["Total Cost"],
                "estimated_rows": result[0]["Plan"]["Plan Rows"]
            }
        except Exception as e:
            logger.error(f"Failed to estimate query cost: {str(e)}")
            return {"estimated_cost": None, "estimated_rows": None}