from WBC import WBC

class ControlFrame:
    def __init__(self, ctrlComp):
        self._ctrlComp = ctrlComp
        self._WBCController = WBC(self._ctrlComp)

    def run(self):
        self._WBCController.run()
