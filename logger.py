import datetime

class ProgramLogger(path, logging):
    def __init__(self, path):
        self.path = path
        self.logging = logging
    if logging:
        def write(message: str):
            time = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with open(path, mode='a') as logfile:
                print(time, message, sep='', file=file)
    else:
        def write(message: str):
            pass
