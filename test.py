import os
import fnmatch
from pathlib import PurePath
def pathgen(path: str, university: str) -> [[]]:

    requirements = ['PKLotSegmented']
    pklot = {}

    if university is not None:
        requirements.append(university)

    for root, _, files in os.walk(top=path, followlinks=True):

        root_path = PurePath(root).parts

        if len(files) > 0:
            if all(requirement in root_path for requirement in requirements):
                for file in files:
                    if fnmatch.fnmatch(file, '*.jpg'):

                        university = root_path[-4]
                        date = root_path[-2]
                        state = root_path[-1]
                        file_path = os.path.abspath(os.path.join(root, file))

                        if university in pklot:
                            if date in pklot[university]:
                                if state in pklot[university][date]:
                                    pklot[university][date][state].append(
                                        file_path)
                                else:
                                    pklot[university][date][state] = [
                                        file_path]
                            else:
                                pklot[university][date] = {state: [file_path]}
                        else:
                            pklot |= {university: {date: {state: [file_path]}}}

    return pklot
