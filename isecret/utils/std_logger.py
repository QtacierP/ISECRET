import sys
import isecret.utils.distributed as du

class StdLog():
    '''
    The logger which prints to both terminal and file
    '''
    def __init__(self, file=None):
        self.terminal = sys.stdout
        self.io_stream = open(file, "w")

    def write(self, message):
        if not du.is_master_proc():
            return 
        self.terminal.write(message)
        self.io_stream.write(message)

    def flush(self):
        pass # TODO
