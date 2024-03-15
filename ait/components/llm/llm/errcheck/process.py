from initial import *


def process_error_check(args) -> None:    
    logger.info("Environment configuring...")
    
    logger.info("User inputs verifying...")
    handles_check_type()
    handles_output_dir()
    handles_exit_flag()
    logger.info("User inputs verified.")
    
    logger.info("Dependencies checking...")
    handles_so_dir()
    logger.info("Dependencies founded.")
    
    logger.info("Environment configuration finished. Inference processing...") 
    handles_exec()
    logger.info("Inference finished.")
    logger.info("Results are stored under the directory: %s.", os.environ['ATB_OUTPUT_DIR'])
