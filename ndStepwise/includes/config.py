from includes.logging import setup_logger
import atexit
class Config:
    def __init__(self, dataset):
        self.settings = {}
        self.model_path = 'C:\\Users\\maxdi\\OneDrive\\Documents\\uni_honours\\models'
        self.log = setup_logger(dataset)
        atexit.register(self.cleanup_logging)

    def cleanup_logging(self):
        # Close and remove all handlers
        for handler in self.log.handlers[:]:
            handler.close()
            self.log.removeHandler(handler)