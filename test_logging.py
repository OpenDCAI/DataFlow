import logging
logging.basicConfig(level=logging.DEBUG,
    format="%(asctime)s | %(pathname)s - %(funcName)s - %(lineno)d - %(module)s - %(name)s | %(levelname)s | Processno %(process)d - Threadno %(thread)d : %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S"
    )
def main():
    
    logging.debug("This is DEBUG message")
    logging.info("This is INFO message")
    logging.warning("This is WARNING message")
    logging.error("This is ERROR message")
    
    return

main()