"""Main entry point for the Assessment Pipeline."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from src.pipelines.assessment_pipeline import AssessmentPipeline
from src.pipelines.base import PipelineContext, PipelineStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Run the assessment pipeline."""
    try:
        logger.info("Starting Assessment Pipeline")
        
        # Create pipeline
        pipeline = AssessmentPipeline()
        
        # Run pipeline
        context = await pipeline.run()
        
        if context.status == PipelineStatus.SUCCESS:
            logger.info("Assessment Pipeline completed successfully")
            return 0
        else:
            logger.error(f"Assessment Pipeline failed with status: {context.status}")
            return 1
        
    except Exception as e:
        logger.error(f"Assessment Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)