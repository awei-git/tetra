"""Base classes for pipeline framework"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from enum import Enum
import asyncio
import traceback

from src.utils.logging import logger


class PipelineStatus(str, Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


@dataclass
class PipelineContext:
    """Context passed between pipeline steps"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: PipelineStatus = PipelineStatus.PENDING
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str):
        """Add error to context"""
        self.errors.append(f"[{datetime.now().isoformat()}] {error}")
        
    def add_warning(self, warning: str):
        """Add warning to context"""
        self.warnings.append(f"[{datetime.now().isoformat()}] {warning}")
        
    def set_metric(self, key: str, value: Any):
        """Set a metric value"""
        self.metrics[key] = value
        
    def increment_metric(self, key: str, value: int = 1):
        """Increment a metric counter"""
        current = self.metrics.get(key, 0)
        self.metrics[key] = current + value
        
    def finish(self, status: PipelineStatus):
        """Mark pipeline as finished"""
        self.end_time = datetime.now()
        self.status = status
        self.metrics["duration_seconds"] = (self.end_time - self.start_time).total_seconds()


T = TypeVar('T')


class PipelineStep(ABC, Generic[T]):
    """Base class for pipeline steps"""
    
    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description or name
        
    @abstractmethod
    async def execute(self, context: PipelineContext) -> T:
        """Execute the pipeline step"""
        pass
        
    async def validate(self, context: PipelineContext) -> bool:
        """Validate if step can be executed (optional override)"""
        return True
        
    async def cleanup(self, context: PipelineContext):
        """Cleanup after step execution (optional override)"""
        pass


class Pipeline(ABC):
    """Base class for data pipelines"""
    
    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description or name
        self.steps: List[PipelineStep] = []
        self.context: Optional[PipelineContext] = None
        
    def add_step(self, step: PipelineStep) -> "Pipeline":
        """Add a step to the pipeline"""
        self.steps.append(step)
        return self
        
    def add_steps(self, steps: List[PipelineStep]) -> "Pipeline":
        """Add multiple steps to the pipeline"""
        self.steps.extend(steps)
        return self
        
    @abstractmethod
    async def setup(self) -> PipelineContext:
        """Setup pipeline context"""
        pass
        
    async def teardown(self, context: PipelineContext):
        """Teardown pipeline (optional override)"""
        pass
        
    async def on_step_success(self, step: PipelineStep, context: PipelineContext, result: Any):
        """Hook called after successful step execution"""
        pass
        
    async def on_step_failure(self, step: PipelineStep, context: PipelineContext, error: Exception):
        """Hook called after step failure"""
        pass
        
    async def run(self, **kwargs) -> PipelineContext:
        """Run the pipeline"""
        logger.info(f"Starting pipeline: {self.name}")
        
        # Setup context
        try:
            self.context = await self.setup()
            self.context.data.update(kwargs)
        except Exception as e:
            logger.error(f"Pipeline setup failed: {e}")
            context = PipelineContext()
            context.add_error(f"Setup failed: {str(e)}")
            context.finish(PipelineStatus.FAILED)
            return context
            
        # Execute steps
        failed_steps = 0
        successful_steps = 0
        
        for i, step in enumerate(self.steps):
            logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
            
            try:
                # Validate step
                if not await step.validate(self.context):
                    logger.warning(f"Step validation failed: {step.name}")
                    self.context.add_warning(f"Step {step.name} skipped due to validation failure")
                    continue
                    
                # Execute step
                result = await step.execute(self.context)
                await self.on_step_success(step, self.context, result)
                successful_steps += 1
                logger.info(f"Step completed successfully: {step.name}")
                
            except Exception as e:
                failed_steps += 1
                error_msg = f"Step {step.name} failed: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                self.context.add_error(error_msg)
                
                # Call failure hook
                await self.on_step_failure(step, self.context, e)
                
                # Decide whether to continue
                if not self.should_continue_on_error(step, e):
                    break
                    
            finally:
                # Cleanup
                try:
                    await step.cleanup(self.context)
                except Exception as e:
                    logger.warning(f"Step cleanup failed: {e}")
                    
        # Determine final status
        if failed_steps == 0:
            status = PipelineStatus.SUCCESS
        elif successful_steps == 0:
            status = PipelineStatus.FAILED
        else:
            status = PipelineStatus.PARTIAL
            
        self.context.finish(status)
        self.context.set_metric("steps_total", len(self.steps))
        self.context.set_metric("steps_successful", successful_steps)
        self.context.set_metric("steps_failed", failed_steps)
        
        # Teardown
        try:
            await self.teardown(self.context)
        except Exception as e:
            logger.error(f"Pipeline teardown failed: {e}")
            self.context.add_error(f"Teardown failed: {str(e)}")
            
        logger.info(
            f"Pipeline completed: {self.name} - "
            f"Status: {self.context.status}, "
            f"Duration: {self.context.metrics.get('duration_seconds', 0):.2f}s, "
            f"Steps: {successful_steps}/{len(self.steps)} successful"
        )
        
        return self.context
        
    def should_continue_on_error(self, step: PipelineStep, error: Exception) -> bool:
        """Determine if pipeline should continue after error (override for custom logic)"""
        return True  # Default: continue on error
        

class ConditionalStep(PipelineStep[T]):
    """A step that executes conditionally based on a predicate"""
    
    def __init__(
        self, 
        step: PipelineStep[T], 
        condition: Callable[[PipelineContext], bool],
        name: Optional[str] = None
    ):
        super().__init__(name or f"Conditional({step.name})")
        self.step = step
        self.condition = condition
        
    async def execute(self, context: PipelineContext) -> Optional[T]:
        if self.condition(context):
            return await self.step.execute(context)
        else:
            logger.info(f"Skipping step {self.step.name} due to condition")
            return None
            

class ParallelStep(PipelineStep[List[Any]]):
    """A step that executes multiple steps in parallel"""
    
    def __init__(self, steps: List[PipelineStep], name: Optional[str] = None):
        super().__init__(name or f"Parallel({len(steps)} steps)")
        self.steps = steps
        
    async def execute(self, context: PipelineContext) -> List[Any]:
        """Execute all steps in parallel"""
        tasks = [step.execute(context) for step in self.steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        if exceptions:
            logger.error(f"Parallel execution had {len(exceptions)} failures")
            for i, exc in enumerate(exceptions):
                if isinstance(exc, Exception):
                    context.add_error(f"Parallel step {i} failed: {str(exc)}")
                    
        return results