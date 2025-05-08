from dataflow.utils.utils import get_logger
def main():
    logger = get_logger()
    logger.debug("This is DEBUG message")
    logger.info("This is INFO message")
    logger.warning("This is WARNING message")
    logger.error("This is ERROR message")
    
    return

main()