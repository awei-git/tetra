import asyncio
import time
from typing import Optional, Dict, Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.logging import logger


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, calls: int, period: int):
        """
        Initialize rate limiter
        
        Args:
            calls: Number of calls allowed
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self.call_times: list[float] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        async with self._lock:
            now = time.time()
            
            # Remove old calls outside the period window
            self.call_times = [
                t for t in self.call_times 
                if now - t < self.period
            ]
            
            # If we've made too many calls, wait
            if len(self.call_times) >= self.calls:
                sleep_time = self.period - (now - self.call_times[0]) + 0.1
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    # Recurse to re-check
                    return await self.acquire()
            
            # Record this call
            self.call_times.append(now)


class BaseAPIClient:
    """Base class for API clients"""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
        timeout: int = 30,
    ):
        """
        Initialize API client
        
        Args:
            base_url: Base URL for API
            api_key: API key for authentication
            rate_limiter: Rate limiter instance
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers=self._get_headers(),
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._client:
            await self._client.aclose()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for requests"""
        headers = {
            "User-Agent": "Tetra Trading Platform/1.0",
            "Accept": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json: JSON body
            
        Returns:
            Response data
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        # Apply rate limiting
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        logger.debug(f"{method} {url}")
        
        try:
            response = await self._client.request(
                method=method,
                url=url,
                params=params,
                json=json,
            )
            
            response.raise_for_status()
            
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make GET request"""
        return await self._request("GET", endpoint, params=params)
    
    async def post(self, endpoint: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make POST request"""
        return await self._request("POST", endpoint, json=json)
    
    async def put(self, endpoint: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make PUT request"""
        return await self._request("PUT", endpoint, json=json)
    
    async def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make DELETE request"""
        return await self._request("DELETE", endpoint)