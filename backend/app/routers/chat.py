from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
import logging
import json
from datetime import datetime

from ..services.database import get_db_session
from ..services.llm import LLMService
from ..services.query_executor import QueryExecutor
from ..models.requests import ChatQueryRequest, SaveQueryRequest
from ..models.responses import ChatQueryResponse, QueryHistoryResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory history for now (should use database in production)
query_history: List[dict] = []


@router.post("/query", response_model=ChatQueryResponse)
async def process_query(
    request: ChatQueryRequest,
    db=Depends(get_db_session)
):
    """Process a natural language query and return appropriate response"""
    try:
        # Initialize services
        llm_service = LLMService()
        query_executor = QueryExecutor(db)
        
        # Generate SQL from natural language
        sql = await llm_service.generate_sql(request.query)
        logger.info(f"Generated SQL: {sql}")
        
        # Validate SQL (read-only)
        if not query_executor.is_safe_query(sql):
            logger.warning(f"Unsafe SQL rejected: {sql}")
            raise HTTPException(
                status_code=400,
                detail="Only SELECT queries are allowed"
            )
        
        # Execute query
        results, columns = await query_executor.execute_query(sql)
        
        # Generate analysis if results exist
        analysis = None
        if results:
            analysis = await llm_service.analyze_results(
                query=request.query,
                results=results[:10],  # Analyze sample
                columns=columns
            )
        
        return ChatQueryResponse(
            sql=sql,
            data=results,
            columns=columns,
            analysis=analysis,
            row_count=len(results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@router.get("/history", response_model=List[QueryHistoryResponse])
async def get_query_history(limit: int = 20):
    """Get recent query history"""
    return query_history[-limit:]


@router.post("/save")
async def save_query(request: SaveQueryRequest):
    """Save a query to history"""
    history_item = {
        "query": request.query,
        "sql": request.sql,
        "timestamp": request.timestamp or datetime.utcnow(),
        "row_count": len(request.results) if request.results else 0
    }
    query_history.append(history_item)
    
    # Keep only last 100 items
    if len(query_history) > 100:
        query_history.pop(0)
    
    return {"status": "saved"}


@router.post("/validate")
async def validate_sql(sql: str):
    """Validate SQL query without executing"""
    try:
        query_executor = QueryExecutor(None)
        is_safe = query_executor.is_safe_query(sql)
        
        return {
            "valid": is_safe,
            "message": "Query is safe to execute" if is_safe else "Only SELECT queries allowed"
        }
    except Exception as e:
        return {
            "valid": False,
            "message": str(e)
        }