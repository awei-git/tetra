#!/bin/bash
# Run the Assessment Pipeline (Stage 4)

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="/tmp"
LOG_FILE="${LOG_DIR}/tetra_assessment_pipeline_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log messages
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check prerequisites
check_prerequisites() {
    log "${YELLOW}Checking prerequisites...${NC}"
    
    # Check if virtual environment exists
    if [ ! -d "$PROJECT_ROOT/.venv" ]; then
        log "${RED}Error: Virtual environment not found. Run 'uv sync' first.${NC}"
        exit 1
    fi
    
    # Check if metrics data exists
    if [ ! -d "$PROJECT_ROOT/data/metrics" ]; then
        log "${RED}Error: Metrics data not found. Run Metrics Pipeline first.${NC}"
        exit 1
    fi
    
    # Check if PostgreSQL is running
    if ! docker ps | grep -q "tetra-postgres"; then
        log "${YELLOW}Warning: PostgreSQL container not running. Database save will be skipped.${NC}"
        DB_AVAILABLE=false
    else
        DB_AVAILABLE=true
    fi
    
    log "${GREEN}Prerequisites check passed.${NC}"
}

# Function to run assessment pipeline
run_assessment() {
    log "${YELLOW}Starting Assessment Pipeline...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Parse command line arguments
    STRATEGIES="all"
    SCENARIOS="all"
    SYMBOLS=""
    SAVE_DB="true"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --strategies)
                STRATEGIES="$2"
                shift 2
                ;;
            --scenarios)
                SCENARIOS="$2"
                shift 2
                ;;
            --symbols)
                SYMBOLS="$2"
                shift 2
                ;;
            --no-db)
                SAVE_DB="false"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log "${RED}Unknown option: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Build Python command
    PYTHON_CMD=".venv/bin/python -m src.pipelines.assessment_pipeline.main"
    
    if [ "$STRATEGIES" != "all" ]; then
        PYTHON_CMD="$PYTHON_CMD --strategies $STRATEGIES"
    fi
    
    if [ "$SCENARIOS" != "all" ]; then
        PYTHON_CMD="$PYTHON_CMD --scenarios $SCENARIOS"
    fi
    
    if [ -n "$SYMBOLS" ]; then
        PYTHON_CMD="$PYTHON_CMD --symbols $SYMBOLS"
    fi
    
    if [ "$SAVE_DB" = "false" ] || [ "$DB_AVAILABLE" = "false" ]; then
        PYTHON_CMD="$PYTHON_CMD --no-db"
    fi
    
    # Run the pipeline
    log "Executing: $PYTHON_CMD"
    if $PYTHON_CMD 2>&1 | tee -a "$LOG_FILE"; then
        log "${GREEN}Assessment Pipeline completed successfully!${NC}"
        return 0
    else
        log "${RED}Assessment Pipeline failed!${NC}"
        return 1
    fi
}

# Function to show help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Run the Assessment Pipeline to evaluate trading strategies.

OPTIONS:
    --strategies <list>  Comma-separated list of strategies to test (default: all)
                        Example: --strategies "buy_and_hold,golden_cross,turtle_trading"
    
    --scenarios <list>   Comma-separated list of scenarios to test (default: all)
                        Example: --scenarios "Historical_Bull_2017,Historical_Bear_2022"
    
    --symbols <list>     Comma-separated list of symbols to test (default: major indices)
                        Example: --symbols "SPY,QQQ,AAPL,MSFT"
    
    --no-db             Skip saving results to database
    
    --help              Show this help message

EXAMPLES:
    # Run all strategies on all scenarios
    $0
    
    # Test specific strategies
    $0 --strategies "buy_and_hold,golden_cross"
    
    # Test on specific symbols
    $0 --symbols "SPY,QQQ,IWM"
    
    # Quick test without database
    $0 --strategies "buy_and_hold" --scenarios "Test_Scenario" --no-db

PREREQUISITES:
    1. Metrics Pipeline must have been run first
    2. PostgreSQL should be running for database storage
    3. Virtual environment must be set up with 'uv sync'

OUTPUT:
    - Results saved to: data/assessment/
    - Rankings report: data/assessment/rankings_report.json
    - Detailed results: data/assessment/detailed_report.json
    - Logs saved to: $LOG_FILE

EOF
}

# Main execution
main() {
    log "${GREEN}=== Tetra Assessment Pipeline ===${NC}"
    log "Log file: $LOG_FILE"
    
    # Check prerequisites
    check_prerequisites
    
    # Run assessment
    if run_assessment "$@"; then
        log "${GREEN}Pipeline execution completed!${NC}"
        
        # Show results location
        log "${YELLOW}Results saved to:${NC}"
        log "  - Assessment results: $PROJECT_ROOT/data/assessment/"
        log "  - Rankings: $PROJECT_ROOT/data/assessment/rankings_report.json"
        log "  - Summary: $PROJECT_ROOT/data/assessment/assessment_pipeline_summary.json"
        log "  - Log file: $LOG_FILE"
        
        exit 0
    else
        log "${RED}Pipeline execution failed!${NC}"
        log "Check log file for details: $LOG_FILE"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"