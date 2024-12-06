from rl_pendulum.ui.main_ui import Main_UI, Main_UI_Callback_Mixin

class UI(Main_UI, Main_UI_Callback_Mixin):
    """ The complete UI/Frontent of this project """
    def __init__(self):
        Main_UI.__init__(self)
        self.set_callbacks() # method of Main_UI_Callback_Mixin to initialize callbacks separately to Main_UI
                             # The inter-UI callbacks should be separated to the Main_UI-only callbacks (Main_UI currently has no Main_UI-only callbacks)