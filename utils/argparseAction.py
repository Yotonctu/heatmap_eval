import argparse
import os

class savedAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        setattr(namespace, 'saved_folder', values)
        if(not os.path.exists(values)):
            try:
                os.makedirs(values)
            except FileExistsError:
                print('file exists')
            


