from includes.logging import setup_logger
import atexit
import uuid
class Config:
    def __init__(self, dataset):
        self.settings = {}
        self.model_path = 'C:\\Users\\maxdi\\OneDrive\\Documents\\uni_honours\\models'
        self.saved_models = "saved_models/"
        self.uuid_current_run = str(uuid.uuid4().hex)
        self.log = setup_logger(dataset)
        atexit.register(self.cleanup_logging)
        self.do_not_save_models = False

    def cleanup_logging(self):
        # Close and remove all handlers
        for handler in self.log.handlers[:]:
            handler.close()
            self.log.removeHandler(handler)
    
    def get_new_uuid(self):
        self.uuid_current_run = str(uuid.uuid4().hex)
    
    def set_do_not_save_models(self):
        self.do_not_save_models = True